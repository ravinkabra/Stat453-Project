# 项目规范

[English](architecture.md) | [中文](architecture_zh.md)

## 文件夹规范

- `_xxx`: 工具类，常用不常改
- `xxx`: 正常业务类
- `xxx_`: 暂用类，临时使用的，或者是实验主要开发的组件

## 文件规范

- `${component}/${type}` 结构：
  - `component.py`: type业务的component的实现类
  - `config.py`: type业务的component实现类的配置类
    - 一般需要包含 `_target_` 用于自动创建
    - `Config` 结尾：component类实例化可直接传入，对应自动实例化方法 `get_class + cls(config)`
    - `Params` 结尾：component类需要将 `_target_` 去掉后，将field转为dict并 `**` 解析，可利用 `instantiate`

