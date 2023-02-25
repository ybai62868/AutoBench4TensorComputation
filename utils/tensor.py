
from typing import List, Optional, Tuple, Sequence, Union, Mapping, Any, Type
import ctypes
import numpy as np

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]
Bool = Union['Expr', bool]

class Node:
    _dispatch_index = {None: 0}

    def __str__(self):
        from hidet.ir.functors.printer import astext
        return astext(self)

    def __repr__(self):
        return str(self)

    def __int__(self):
        return None

    @classmethod
    def class_index(cls):
        if not hasattr(cls, '_class_index'):
            setattr(cls, '_class_index', len(Node._dispatch_index))
            Node._dispatch_index[cls] = getattr(cls, '_class_index')
        return getattr(cls, '_class_index')

    @staticmethod
    def dispatch_table(mapping: Mapping[Type['Node'], Any]) -> List[Any]:
        table = []
        for cls, target in mapping.items():
            idx = cls.class_index()
            while idx >= len(table):
                table.append(None)
            table[idx] = target
        return table

class DataLayout(Node):
    def __init__(self, shape=None, size=None):
        self.shape: Tuple[Int] = tuple([int(v) if isinstance(v, ir.Constant) else v for v in shape]) if shape is not None else None
        self.size: Int = size

    def __call__(self, *args: Int):
        return self.serialize(*args)

    def __add__(self, other):
        return DataLayout.concat(lhs=self, rhs=other)

    def __radd__(self, other):
        return DataLayout.concat(lhs=other, rhs=self)

    def __mul__(self, other):
        return DataLayout.product(outer=self, inner=other)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]

    def global2local(self, *args: Int) -> Int:
        raise NotImplementedError()

    def global2cond(self, *args: Int) -> Bool:
        raise NotImplementedError()

    def serialize(self, *args: Int):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        # var2value = OrderedDict()
        # arg_vars = variablize(args, var2value)
        # scalar_index = self.global2local(*arg_vars)
        # scalar_index = concat_let_expr(var2value=var2value, body=scalar_index)
        scalar_index = self.global2local(*args)
        return scalar_index

    def within_bound(self, *args: Int):
        if isinstance(args[0], (tuple, list)) and len(args) == 1:
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        var2value = OrderedDict()
        arg_vars = variablize(args, var2value)
        cond = self.global2cond(*arg_vars)
        cond = concat_let_expr(var2value=var2value, body=cond)
        return cond

    def tile(self, inner_shape: Sequence[Int]):
        return TiledDataLayout(base=self, inner_shape=inner_shape)

    def split(self, dim2factor: Mapping[int, Int]):
        return SplitDataLayout(base=self, dim2factor=dim2factor)

    def reorder(self, order: Sequence[int]):
        return self.fuse(order)

    def fuse(self, dim2fuse: Sequence[Union[Sequence[int], int]]):
        return FusedDataLayout(base=self, dim2fuse=dim2fuse)

    def slice_out(self, dims: Sequence[int]):
        return SliceOutDataLayout(base=self, dims=dims)

    @staticmethod
    def product(outer, inner):
        return ProductDataLayout(outer, inner)

    @staticmethod
    def concat(lhs, rhs):
        lhs = to_data_layout(lhs)
        rhs = to_data_layout(rhs)
        return ConcatDataLayout(lhs, rhs)

    @staticmethod
    def local(shape: Sequence[Int]):
        return LocalLayout(shape=shape)

    @staticmethod
    def row_major(shape: Sequence[Int]):
        return RowMajorLayout(shape)

    @staticmethod
    def column_major(shape: Sequence[Int]):
        return ColumnMajorLayout(shape)



class Storage:

    def __init__(self, device, addr, num_bytes, free_handler):
        self.device: str = device
        self.addr: int = addr
        self.num_bytes: int = num_bytes
        self.free_handler: Callable[[Storage], None] = free_handler

    def __del__(self):
        if self.addr != 0:
            self.free_handler(self)

    def __getstate__(self):
        raise ValueError()

    def __setstate__(self, state):
        raise ValueError()

    def cpu(self):
        if self.device == 'cpu':
            return self
        elif self.device == 'cuda':
            host_storage = self.new('cpu', self.num_bytes)
            cuda.memcpy_async(src_addr=self.addr, dst_addr=host_storage.addr, num_bytes=self.num_bytes, kind=cuda.DeviceToHost)
            return host_storage
        else:
            raise NotImplementedError()

    def cuda(self):
        if self.device == 'cuda':
            return self
        elif self.device == 'cpu':
            cuda_storage = self.new('cuda', self.num_bytes)
            cuda.memcpy_async(src_addr=self.addr, dst_addr=cuda_storage.addr, num_bytes=self.num_bytes, kind=cuda.HostToDevice)
            return cuda_storage
        else:
            raise NotImplementedError()

    @staticmethod
    def new(device: str, num_bytes: int) -> 'Storage':
        if device == 'cpu':
            return CpuMemoryPool.current().allocate(nbytes=num_bytes)
        elif device == 'cuda':
            return CudaMemoryPool.current().allocate(nbytes=num_bytes)
        else:
            raise ValueError("Unrecognized device '{}', candidates: {}".format(device, ['cpu', 'cuda']))

    def as_array(self, num_elements: int, dtype: str = 'float32') -> np.ndarray:
        """
        Convert to one-dimension numpy array, sharing the underlying storage.
        Parameters
        ----------
        num_elements: int
            The number of elements in the array. Because the storage may have a larger allocated memory, we can not
            infer the desired number of elements.
        dtype: str, default 'float32'
            The type of data in this storage.
        Returns
        -------
        ret: numpy.ndarray
            A numpy ndarray with one dimension that share the same data as the storage.
        """
        dtype2ctype = {
            'float32': ctypes.c_float,
            'float16': ctypes.c_uint16,
            'int32': ctypes.c_int32,
            'int64': ctypes.c_int64,
            'bool': ctypes.c_bool
        }
        dtype2nptype = {
            'float16': np.float16
        }

        if self.device != 'cpu':
            raise ValueError('The storage must be cpu storage. Please use .cpu() to convert first.')
        buf = (dtype2ctype[dtype] * num_elements).from_address(self.addr)
        buf._hidet_storage = self  # so this storage will not be freed as long as the buffer not been freed.
        assert ctypes.sizeof(buf) <= self.num_bytes, 'Trying to view a storage as a larger array'
        with warnings.catch_warnings():
            # temporarily ignore a warning due to python bug.
            # See: https://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array
            warnings.simplefilter('ignore')
            array = np.ctypeslib.as_array(buf)
        if dtype in dtype2nptype:
            # reinterpret the array when needed
            array = array.view(dtype2nptype[dtype])
        return array


class Tensor:
    def __init__(self,
                 shape: Sequence[int],
                 dtype: str,
                 device: str,
                 storage: Optional[Storage],
                 layout: DataLayout = None,
                 trace: Optional[Tuple['Operator', int]] = None):
        from hidet.tos.operator import Operator
        self.shape = [int(v) for v in shape]
        self.dtype = str(dtype)
        self.device = device
        self.storage = storage
        self.layout = layout if layout else DataLayout.row_major(shape)
        self.trace: Optional[Tuple[Operator, int]] = trace

    def __neg__(self) -> Tensor:
        from .ops import neg
        return neg(self)

    def __add__(self, other) -> Tensor:
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        from .ops import add
        return add(other, self)

    def __sub__(self, other) -> Tensor:
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other) -> Tensor:
        from .ops import multiply
        return multiply(self, other)

    def __rmul__(self, other):
        from .ops import multiply
        return multiply(other, self)

    def __truediv__(self, other) -> Tensor:
        from .ops import divide
        return divide(self, other)

    def __str__(self):
        head = self.signature()
        if self.storage:
            array_str = str(self.cpu().numpy())
            return '{}\n{}'.format(head, array_str)
        else:
            return head + ' with empty storage'

    def __getitem__(self, item):
        from hidet.tos.ops import strided_slice
        if not isinstance(item, tuple):
            item = tuple([item])
        rank = len(self.shape)
        if all(not isinstance(v, slice) for v in item) and len(item) == rank:
            # element access
            return strided_slice(self, starts=list(item), ends=[v + 1 for v in item]).numpy().flatten()[0]
        else:
            while len(item) < rank:
                item = item + (slice(None, None, None),)
            starts, ends, steps = [], [], []
            squeeze_dims = []
            for dim, v in enumerate(item):
                if isinstance(v, int):
                    squeeze_dims.append(dim)
                    starts.append(v)
                    ends.append(v + 1)
                    steps.append(1)
                else:
                    assert isinstance(v, slice)
                    starts.append(v.start if v.start is not None else 0)
                    ends.append(v.stop if v.stop is not None else self.shape[dim])
                    steps.append(v.step if v.step is not None else 1)
            sliced = strided_slice(self, starts, ends, strides=steps).squeeze(squeeze_dims)
            return sliced

    def __iter__(self):
        raise TypeError('hidet.Tensor does not support iteration.')

    def __getstate__(self):
        if self.storage:
            data = self.detach().numpy()
        else:
            data = None

        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'data': data,
            'layout': self.layout,
            'trace': self.trace
        }

    def __setstate__(self, state):
        data = state['data']
        if data is not None:
            assert isinstance(data, np.ndarray)
            tensor = from_numpy(data)
            if state['device'] == 'cuda':
                tensor = tensor.cuda()
            storage = tensor.storage
        else:
            storage = None

        self.shape = state['shape']
        self.dtype = state['dtype']
        self.device = state['device']
        self.storage = storage
        self.layout = state['layout']
        self.trace = state['trace']

    def signature(self) -> str:
        return "Tensor(shape={}, dtype='{}', device='{}')".format(self.shape, self.dtype, self.device)

    @property
    def nbytes(self):
        return prod(self.shape) * dtype_bytes(self.dtype)

    @property
    def op(self):
        return self.trace[0] if self.trace else None

    def scalar(self) -> Union[float, int]:
        if len(self.shape) != 0:
            raise ValueError('Can not convert a Tensor with shape {} to a scalar.'.format(self.shape))
        value = self.numpy().tolist()
        assert isinstance(value, (int, float))
        return value

    def contiguous(self):
        if isinstance(self.layout, RowMajorLayout):
            return self
        return self.reshape(self.shape)

    def reshape(self, shape: Sequence[int]):
        from .ops import reshape
        return reshape(self, shape)

    def squeeze(self, dims: Union[int, Sequence[int]]):
        from .ops import squeeze
        return squeeze(self, dims)

    def unsqueeze(self, dims: Union[int, Sequence[int]]):
        from .ops import unsqueeze
        return unsqueeze(self, dims)

    def rearrange(self, plan: List[List[int]]):
        from .ops import rearrange
        return rearrange(self, plan)

    def flatten(self, start_dim=0, end_dim=None):
        from .ops import flatten
        return flatten(self, start_dim, end_dim)

    def transpose(self, axes: Optional[Sequence[int]]):
        from .ops import transpose
        return transpose(self, axes)

    def barrier(self) -> Tensor:
        from .ops import barrier
        return barrier(self)

    def sum(self, dims: Union[int, List[int]], keep_dim: bool = False):
        from .ops import reduce_sum
        return reduce_sum(self, dims=dims, keep_dim=keep_dim)

    def mean(self, dims: Union[int, List[int]], keep_dim: bool = False):
        from .ops import reduce_mean
        return reduce_mean(self, dims=dims, keep_dim=keep_dim)

    def rsqrt(self):
        from .ops import rsqrt
        return rsqrt(self)

    def cast(self, dtype):
        from .ops import cast
        return cast(self, dtype)

    def cpu(self):
        if self.device == 'cpu':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda(self):
        if self.device == 'cuda':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cuda', self.storage.cuda() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def detach(self):
        if self.trace is None:
            return self
        else:
            return Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                storage=self.storage,
                layout=self.layout,
                trace=None
            )

    def numpy(self) -> np.ndarray:
        if self.device != 'cpu':
            return self.cpu().numpy()
        # convert if this tensor is not in row major layout
        storage = self.contiguous().storage

        # because numpy does not support bfloat16, we convert it into float32
        if self.dtype == 'bfloat16':
            return self.cast('float32').numpy()
        else:
            array = storage.as_array(num_elements=prod(self.shape), dtype=self.dtype)
            return array.reshape(self.shape)



if __name__ == "__main__":
	tensor_test = Tensor()
	print("ojbk")

