# Project Specifications

[English](architecture.md) | [中文](architecture_zh.md)

## Folder Specifications

- `_xxx`: Utility classes, commonly used and relatively stable
- `xxx`: Core business classes
- `xxx_`: Temporary classes, for temporary use

## File Specifications

- `${component}/${type}` structure:
  - `component.py`: Implementation class for ${type} type business component
  - `config.py`: Configuration class for ${type} type business component implementation
    - Usually needs to include `_target_` field for automatic instantiation
    - `Config` suffix: Component class can be passed directly during instantiation, corresponding to automatic instantiation method `get_class + cls(config)`
    - `Params` suffix: Configuration parameter class that supports direct component instantiation via `hydra.instantiate`, without manual field conversion
