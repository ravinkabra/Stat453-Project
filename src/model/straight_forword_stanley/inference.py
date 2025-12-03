from src.model.straight_forword_stanley.model import (
    OldPtSFSTransformer,
    EOS_TOKEN,
    PAD_TOKEN,
    SOM_TOKEN,
    EOM_TOKEN,
    SOA_TOKEN,
    EOA_TOKEN,
)
from src._utils.preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES
import torch
import pretty_midi
import os
import argparse
from typing import Literal
import time

# === C-major normalization: ADDED ===
# Key detection and C-major conversion helpers for inference.
# We detect the (major) key of the melody, compute the semitone shift
# to move it to C major, transpose both melody and accompaniment,
# and later transpose the generated C-major accompaniment back
# to the original key.

C_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33,
                   4.38, 4.09, 2.52, 5.19,
                   2.39, 3.66, 2.29, 2.88]

KEY_NAMES_MAJOR = ["C", "C#", "D", "Eb", "E", "F",
                   "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_major_key(pm: pretty_midi.PrettyMIDI):
    """Estimate major key (0=C,1=C#,...) and semitone shift to C.

    Returns:
        best_root (int): 0-11, where 0 means C major.
        best_name (str): Human-readable major key name.
        shift_to_c (int): Semitone shift so that adding shift_to_c
            moves the piece into C major. Negative means transpose down.
    """
    pc_hist = [0.0] * 12
    for inst in pm.instruments:
        for note in inst.notes:
            pc = note.pitch % 12
            duration = float(note.end - note.start)
            if duration <= 0.0:
                duration = 0.1
            pc_hist[pc] += duration

    if sum(pc_hist) == 0.0:
        return 0, "C", 0

    best_score = None
    best_root = 0
    for root in range(12):
        score = 0.0
        for i in range(12):
            pc = (root + i) % 12
            score += pc_hist[pc] * C_MAJOR_PROFILE[i]
        if best_score is None or score > best_score:
            best_score = score
            best_root = root

    shift_to_c = (-best_root) % 12
    if shift_to_c > 6:
        # Prefer small negative shifts instead of large positives
        shift_to_c -= 12

    return best_root, KEY_NAMES_MAJOR[best_root], shift_to_c


def make_c_major_pair(mel_path: str, inference_root: str):
    """Create a C-major copy of the melody and its corresponding acc.

    The original files are not modified. New files are written under
    `inference_root/c_major/...` with the same relative paths.

    Returns:
        c_mel_path, c_acc_path, key_name, shift_to_c
    """
    acc_path = mel_path.replace("mel", "acc")
    if not os.path.isfile(acc_path):
        raise FileNotFoundError(f"Accompaniment file not found: {acc_path}")

    pm_mel = pretty_midi.PrettyMIDI(mel_path)
    root, key_name, shift_to_c = estimate_major_key(pm_mel)
    print(f"[C-major] {mel_path}: detected key {key_name}, shift_to_c={shift_to_c:+d}")

    rel_mel = os.path.relpath(mel_path)
    rel_acc = os.path.relpath(acc_path)
    c_mel_path = os.path.join(inference_root, "c_major", rel_mel)
    c_acc_path = os.path.join(inference_root, "c_major", rel_acc)

    os.makedirs(os.path.dirname(c_mel_path), exist_ok=True)
    os.makedirs(os.path.dirname(c_acc_path), exist_ok=True)

    # Transpose melody to C major
    for inst in pm_mel.instruments:
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift_to_c))
    pm_mel.write(c_mel_path)

    # Transpose accompaniment with same shift
    pm_acc = pretty_midi.PrettyMIDI(acc_path)
    for inst in pm_acc.instruments:
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift_to_c))
    pm_acc.write(c_acc_path)

    return c_mel_path, c_acc_path, key_name, shift_to_c


def decode_output(outputs, save_path, tempo=120.0, prompt=True, single=False, transpose_semitones: int = 0):
    """将模型输出的 token 序列解码为 MIDI 文件。

    Args:
        outputs: 模型生成的输出序列，可以是单个序列或元组
        save_path: MIDI 文件保存路径
        tempo: 音乐速度（BPM）
        prompt: 是否包含 prompt 部分
        single: 是否为单一轨道（仅旋律）
        transpose_semitones: === C-major normalization: ADDED ===
            在写出之前，对整首曲子进行整体移调（半音数）。
            这里用于将 C 大调输出转换回原始输入的调。
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for output in outputs:
        instrument_map: dict[Literal[0, 1], pretty_midi.Instrument] = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            time_step = time_step if single else time_step // 2
            start_time = time_step * time_step_length

            for i in range(0, len(content), 2):
                program = int(content[i].item())

                # 跳过特殊 token
                if program in [EOS_TOKEN, PAD_TOKEN, SOM_TOKEN, EOM_TOKEN, SOA_TOKEN, EOA_TOKEN]:
                    if program == EOS_TOKEN:
                        break
                    continue

                if i + 1 >= len(content):
                    print("Incomplete note @", time_step, i)
                    break

                pitch_duration = int(content[i + 1].item())
                pitch_duration = pitch_duration - 2
                pitch = pitch_duration % 128
                duration = pitch_duration // 128

                if program != 0 and program != 1:
                    print("Invalid program:", program, "@", time_step, i)
                    break

                if pitch < 0 or pitch >= 128:
                    print("Invalid pitch:", pitch, "@", time_step, i)
                    break

                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print("Invalid duration:", duration, "@", time_step, i)
                    break

                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time

                if program not in instrument_map:
                    if program == 0:
                        inst = pretty_midi.Instrument(program=24, name="Guitar")
                    else:  # program == 1
                        inst = pretty_midi.Instrument(program=0, name="Piano")
                    instrument_map[program] = inst
                    midi.instruments.append(inst)

                inst = instrument_map[program]
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))

    # === C-major normalization: ADDED ===
    # Optionally transpose the decoded MIDI (e.g. from C major back to the original key).
    if transpose_semitones != 0:
        for inst in midi.instruments:
            for note in inst.notes:
                note.pitch = max(0, min(127, note.pitch + transpose_semitones))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def decompress(model, byte_arr_mel, byte_arr_acc, device):
    """将预处理的 MIDI 数据转换为模型输入格式。

    Args:
        model: 模型实例
        byte_arr_mel: 旋律数据的字节数组
        byte_arr_acc: 伴奏数据的字节数组
        device: 计算设备

    Returns:
        处理后的旋律和伴奏张量
    """
    x = torch.tensor(byte_arr_mel).unsqueeze(0)
    x = x.to(device)
    y = torch.tensor(byte_arr_acc).unsqueeze(0)
    y = y.to(device)
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).to(device), y=y)


def continuation(
    model, midi_path, prompt_length=100, generation_length=400, temperature=1.0, n_samples=1, device="cuda"
):
    """使用给定的旋律 prompt 生成伴奏。

    Args:
        model: 训练好的 OldPtSFSTransformer 模型
        midi_path: 输入 MIDI 文件路径（旋律）
        prompt_length: prompt 的长度（帧数）
        generation_length: 要生成的伴奏长度（帧数）
        temperature: 采样温度
        n_samples: 生成样本数量
        device: 计算设备
    """
    if not os.path.isfile(midi_path):
        print(f"Error: {midi_path} is not a valid file.")
        return

    # === C-major normalization: ADDED ===
    # 预处理 MIDI 文件（先转换为 C 大调再送入模型）
    inference_root = "temp_cmajor"
    c_mel_path, c_acc_path, key_name, shift_to_c = make_c_major_pair(midi_path, inference_root)
    print(f"[C-major] Inference input key: {key_name}, shift_to_c={shift_to_c:+d}")
    transpose_back = -shift_to_c

    byte_arr_mel = preprocess_midi(c_mel_path, 4)
    byte_arr_acc = preprocess_midi(c_acc_path, 4)

    if byte_arr_mel is None:
        print(f"Error: preprocess_midi returned None for mel file: {c_mel_path}")
        return

    if byte_arr_acc is None:
        print(f"Error: preprocess_midi returned None for acc file: {c_acc_path}")
        return

    # 解压并处理数据（此时所有 token 都对应 C 大调）
    x_mel, x_acc = decompress(model, byte_arr_mel[0], byte_arr_acc[0], device=device)

    # 提取 prompt 部分
    x_mel_prompt = x_mel[:, :prompt_length]

    # 保存原始旋律用于对比（这里保存的是 C 大调版本，方便调试）
    decode_output(
        [x_mel[:, i, :] for i in range(x_mel.shape[1])],
        f"temp/{model.save_name}/{os.path.basename(midi_path)}_original_melody.mid",
        single=True,
        tempo=90.0,
        transpose_semitones=0,
    )

    # 保存 prompt 部分（C 大调）
    if prompt_length != 0:
        decode_output(
            [x_mel_prompt[:, i, :] for i in range(x_mel_prompt.shape[1])],
            f"temp/{model.save_name}/{os.path.basename(midi_path)}_prompt_len{prompt_length}.mid",
            single=True,
            tempo=90.0,
            transpose_semitones=0,
        )

    # 开始生成
    with torch.no_grad():
        x_mel_prompt = x_mel_prompt.repeat(n_samples, 1, 1)
        x_acc_prompt = x_acc[:, :prompt_length].repeat(n_samples, 1, 1)
        x_acc_prompt_tolist = [
            x_acc_prompt[:, i : i + 1, :].view(-1, x_acc_prompt.shape[-1]) for i in range(x_acc_prompt.shape[1])
        ]

        start_time = time.time()

        # 使用模型的 global_sampling 方法生成伴奏
        output = model.global_sampling(x_mel=x_mel_prompt, max_len=generation_length, temperature=temperature)
        output = x_acc_prompt_tolist + output  # 将 prompt 和生成的伴奏合并

        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f} seconds")

    # 保存生成结果
    for i in range(n_samples):
        # output 是 [B, generated_len, L]
        # 需要转换为列表格式以便 decode_output 处理
        output_i = [output[j][i : i + 1, :] for j in range(len(output))]

        # 保存为 MIDI（这里从 C 大调移调回原始调）
        decode_output(
            output_i,
            f"temp2/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_{i}.mid",
            single=True,  # 只有伴奏
            tempo=90.0,
            transpose_semitones=transpose_back,
        )

        # 保存为张量（可选，用于调试）
        with open(f"temp/{model.save_name}/tensor_output_{i}.txt", "w") as f:
            f.write(str(output_i))


if __name__ == "__main__":
    start_pre_time = time.time()

    parser = argparse.ArgumentParser(description="OldPtSFSTransformer inference script")

    parser.add_argument(
        "--model_path",
        type=str,
        default="output/straightforward_m2a/sfs_m2a_experiment/0.0.10/checkpoints/epoch=999-step=51000.ckpt",
        help="path to model checkpoint",
    )
    parser.add_argument("--prompt_len", type=int, default=75, help="length of prompt")
    parser.add_argument("--n_samples", type=int, default=1, help="number of samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--generation_length", type=int, default=150, help="number of frames to generate")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_path = args.model_path
    print(f"Loading model from: {model_path}")

    model = OldPtSFSTransformer.load_from_checkpoint(checkpoint_path=model_path, map_location=device)
    model.save_name = os.path.basename(model_path)
    model.to(device)
    model.eval()

    # MIDI 文件路径集合
    midi_file_path_set = [
        "/home/ubuntu/stanleyz/StreamMUSE/inference_benchmark/inputs/aria_unique_skyline_top2_subset_5/mel",
        "/home/ubuntu/stanleyz/StreamMUSE/input/mel",
        "/home/ubuntu/stanleyz/StreamMUSE/inference_benchmark/inputs/test_set/mel",
    ]

    end_pre_time = time.time()
    print(f"Model loading time: {end_pre_time - start_pre_time:.2f} seconds")

    # 遍历所有 MIDI 文件进行推理
    for midi_file_path in midi_file_path_set:
        if not os.path.exists(midi_file_path):
            print(f"Warning: Directory not found: {midi_file_path}")
            continue

        for midi_file in os.listdir(midi_file_path):
            if midi_file.endswith(".mid"):
                midi = os.path.join(midi_file_path, midi_file)
                print(f"\nProcessing: {midi}")

                start_true_generation_time = time.time()
                continuation(
                    model,
                    midi,
                    temperature=args.temperature,
                    generation_length=args.generation_length,
                    n_samples=args.n_samples,
                    prompt_length=args.prompt_len,
                    device=device,
                )
                end_true_generation_time = time.time()

                print(
                    f"Total time for generating {midi}: {end_true_generation_time - start_true_generation_time:.2f} seconds"
                )
