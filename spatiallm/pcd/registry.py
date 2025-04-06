"""
参考: PointCept项目

@misc{pointcept2023,
    title={Pointcept: 点云感知研究的代码库},
    author={Pointcept贡献者},
    year={2023}
}
"""

import inspect  # 用于检查对象类型
import warnings  # 警告处理
from collections import abc  # 抽象基类
from functools import partial  # 偏函数


def is_seq_of(seq, expected_type, seq_type=None):
    """检查是否是某种类型的序列
    
    参数:
        seq (Sequence): 要检查的序列
        expected_type (type): 期望的序列元素类型
        seq_type (type, 可选): 期望的序列类型
    
    返回:
        bool: 序列是否有效
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence  # 默认使用抽象序列类型
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):  # 检查序列类型
        return False
    for item in seq:  # 检查每个元素的类型
        if not isinstance(item, expected_type):
            return False
    return True


def build_from_cfg(cfg, registry, default_args=None):
    """根据配置字典构建模块
    
    参数:
        cfg (dict): 配置字典，至少包含"type"键
        registry (:obj:`Registry`): 注册表对象
        default_args (dict, 可选): 默认初始化参数
    
    返回:
        object: 构建的对象
    
    异常:
        TypeError: 当输入类型不匹配时
        KeyError: 当缺少必要键时
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg必须是字典，但得到{type(cfg)}")
    if "type" not in cfg:  # 检查type键是否存在
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg`或`default_args`必须包含"type"键，'
                f"但得到{cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):  # 检查注册表类型
        raise TypeError(
            "registry必须是Registry对象，" f"但得到{type(registry)}"
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args必须是字典或None，" f"但得到{type(default_args)}"
        )

    args = cfg.copy()  # 复制配置字典

    # 合并默认参数
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)  # 只添加不存在的键

    obj_type = args.pop("type")  # 获取并移除type键
    if isinstance(obj_type, str):  # 如果是字符串类型
        obj_cls = registry.get(obj_type)  # 从注册表获取类
        if obj_cls is None:
            raise KeyError(f"{obj_type}不在{registry.name}注册表中")
    elif inspect.isclass(obj_type):  # 如果是类对象
        obj_cls = obj_type
    else:
        raise TypeError(f"type必须是字符串或有效类型，但得到{type(obj_type)}")

    try:
        return obj_cls(**args)  # 实例化对象
    except Exception as e:
        # 增强错误信息，包含类名
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """一个将字符串映射到类的注册表。

    可以从注册表构建已注册的对象。
    示例:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    高级用法请参考:
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html

    参数:
        name (str): 注册表名称。
        build_func(func, 可选): 从注册表构建实例的函数，
            如果未指定parent或build_func，则使用func:`build_from_cfg`。
            如果指定了parent但未指定build_func，则build_func将从parent继承。
            默认: None。
        parent (Registry, 可选): 父注册表。子注册表中注册的类可以从父注册表构建。
            默认: None。
        scope (str, 可选): 注册表的作用域。它是搜索子注册表的关键字。
            如果未指定，作用域将是定义类的包名，如mmdet、mmcls、mmseg。
            默认: None。
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name  # 注册表名称
        self._module_dict = dict()  # 存储注册类的字典
        self._children = dict()  # 子注册表字典
        # 推断或设置作用域
        self._scope = self.infer_scope() if scope is None else scope  

        # build_func的优先级设置:
        # 1. 直接传入的build_func
        # 2. 父注册表的build_func
        # 3. 默认的build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func  # 继承父注册表的构建函数
            else:
                self.build_func = build_from_cfg  # 使用默认构建函数
        else:
            self.build_func = build_func  # 使用自定义构建函数
            
        # 设置父注册表关系
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)  # 将当前注册表添加为父注册表的子项
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        """返回注册表中类的数量"""
        return len(self._module_dict)

    def __contains__(self, key):
        """检查键是否存在于注册表中"""
        return self.get(key) is not None

    def __repr__(self):
        """注册表的字符串表示"""
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def infer_scope():
        """推断注册表的作用域。

        返回注册表定义所在包的名称。

        示例:
            # 在mmdet/models/backbone/resnet.py中
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            ``ResNet``的作用域将是``mmdet``。

        返回:
            scope (str): 推断出的作用域名称。
        """
        # inspect.stack()追踪此函数的调用位置，索引2表示调用infer_scope()的帧
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]  # 返回包名

    @staticmethod
    def split_scope_key(key):
        """拆分作用域和键名。

        从键名中分离出第一个作用域。

        示例:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        返回:
            scope (str, None): 第一个作用域。
            key (str): 剩余的键名。
        """
        split_index = key.find(".")  # 查找分隔点位置
        if split_index != -1:  # 如果找到分隔点
            return key[:split_index], key[split_index + 1 :]  # 返回作用域和键名
        else:
            return None, key  # 无作用域时返回None和原键名

    @property
    def name(self):
        """获取注册表名称的只读属性"""
        return self._name

    @property
    def scope(self):
        """获取注册表作用域的只读属性"""
        return self._scope

    @property
    def module_dict(self):
        """获取模块字典的只读属性"""
        return self._module_dict

    @property
    def children(self):
        """获取子注册表字典的只读属性"""
        return self._children

    def get(self, key):
        """获取注册表中的记录。

        参数:
            key (str): 字符串格式的类名。

        返回:
            class: 对应的类。

        搜索顺序:
            1. 当前注册表
            2. 子注册表
            3. 根注册表
        """
        scope, real_key = self.split_scope_key(key)  # 拆分作用域和键名
        if scope is None or scope == self._scope:  # 如果无作用域或匹配当前作用域
            # 从当前注册表获取
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # 从子注册表获取
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # 向上查找根注册表
                parent = self.parent
                while parent.parent is not None:  # 循环直到根注册表
                    parent = parent.parent
                return parent.get(key)  # 从根注册表获取

    def build(self, *args, **kwargs):
        """构建注册对象的快捷方法"""
        return self.build_func(*args, **kwargs, registry=self)  # 使用注册表的构建函数

    def _add_children(self, registry):
        """为注册表添加子注册表。

        基于作用域添加子注册表。
        父注册表可以从子注册表构建对象。

        示例:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """
        assert isinstance(registry, Registry)  # 确保是注册表实例
        assert registry.scope is not None  # 确保有作用域
        assert (
            registry.scope not in self.children
        ), f"作用域 {registry.scope} 已存在于 {self.name} 注册表中"
        self.children[registry.scope] = registry  # 添加子注册表
        
        
    def _register_module(self, module_class, module_name=None, force=False):
        """内部注册模块方法
        
        参数:
            module_class: 要注册的模块类
            module_name: 注册名称(可选)
            force: 是否强制覆盖已存在的注册项
            
        异常:
            TypeError: 当module_class不是类时
            KeyError: 当名称已存在且force=False时
        """
        if not inspect.isclass(module_class):
            raise TypeError("module必须是类，" f"但得到{type(module_class)}")

        if module_name is None:  # 默认使用类名
            module_name = module_class.__name__
        if isinstance(module_name, str):  # 支持单个名称或名称列表
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:  # 检查名称冲突
                raise KeyError(f"{name}已注册在{self.name}中")
            self._module_dict[name] = module_class  # 注册类

    def deprecated_register_module(self, cls=None, force=False):
        """已弃用的注册方法(保持向后兼容)
        
        警告:
            此旧API将被移除，请使用新的register_module API
        """
        warnings.warn(
            "旧的register_module(module, force=False) API已弃用并将被移除，"
            "请使用新的register_module(name=None, force=False, module=None) API"
        )
        if cls is None:  # 作为装饰器使用时
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)  # 注册类
        return cls

    def register_module(self, name=None, force=False, module=None):
        """注册模块方法
        
        可以向self._module_dict添加记录，键是类名或指定名称，值是类本身。
        可以作为装饰器或普通函数使用。

        示例:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()  # 作为装饰器
            >>> class ResNet:
            >>>     pass

            >>> @backbones.register_module(name='mnet')  # 指定名称
            >>> class MobileNet:
            >>>     pass

            >>> backbones.register_module(ResNet)  # 直接注册

        参数:
            name (str | None): 注册的模块名称。未指定时使用类名。
            force (bool): 是否覆盖同名注册项，默认False。
            module (type): 要注册的模块类。

        返回:
            注册的类或装饰器函数

        异常:
            TypeError: 当参数类型不匹配时
        """
        if not isinstance(force, bool):
            raise TypeError(f"force必须是布尔值，但得到{type(force)}")
        # 兼容旧API的临时方案(可能引入意外错误)
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # 提前检查参数类型
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name必须是None、字符串或字符串序列，"
                f"但得到{type(name)}"
            )

        # 作为普通方法使用: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # 作为装饰器使用: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register
