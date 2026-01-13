import numpy as np

# Complex floating point
tCpx = np.complex128
# Real floating point
tReal = np.float64

from mpi4py import MPI

import jax

from functools import partial
import collections

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# jax.distributed.initialize() # for use on the cluster
jax.distributed.initialize('localhost:10000', num_processes=size, process_id=rank) # for use on local machine

print(f"Global list of devices on process {rank}: {jax.devices()}")
print(f"Local devices on process {rank}: {jax.local_devices()}")



try:
    myDevice = jax.local_devices()[MPI.COMM_WORLD.Get_rank() % len(jax.local_devices())]
except:
    myDevice = jax.local_devices()[0]
    print("WARNING: Could not assign devices based on MPI ranks. Assigning default device ", myDevice)

myPmapDevices = jax.local_devices()  # [myDevice]
myDeviceCount = len(myPmapDevices)
pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)

# >>>>>>>>>>>>>>>>>>>>>

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

mesh = Mesh(jax.devices(), ('i',))
sharding = NamedSharding(mesh, P('i'))

myShardmapDevices = jax.devices() # shard_map can work with all devices
myDeviceCountAll = len(myShardmapDevices)
mesh = jax.make_mesh((myDeviceCountAll,), ('i',))
shardmap_for_my_devices = partial(jax.shard_map, mesh=mesh)

def visualize_array_sharding(arr):
  if jax.process_index() == 0:
    print("---")
    jax.debug.visualize_array_sharding(arr)

def print_once(msg):
  if jax.process_index() == 0:
    print(msg)

def to_named_sharding(arr):
    """
    Convert a PmapSharding to NamedSharding while merging the device dimension with the batch diension.
    The input array can be sharded across processes and devices.
    There is no data movement between devices and the host.
    """
    arr = jax.make_array_from_process_local_data(sharding, arr) # automatically converts to NamedSharding
    arr = arr.reshape((-1,) + arr.shape[2:]) # remove the device dimension
    return arr

def to_pmap_sharding(arr):
    """
    Convert a NamedSharding array to PmapSharding array while splitting the batch dimension to add the device dimension.
    This involves data movement between devices and the host, which should be avoided for large arrays.
    """
    arr = arr.reshape((myDeviceCountAll, -1,) + arr.shape[1:])
    local_arr = arr[jax.process_index()*myDeviceCount:(jax.process_index()+1)*myDeviceCount]

    # Break the connection to global sharding by going through host memory
    local_arr_host = jax.device_get(local_arr)

    # Redistribute it on local devices
    local_arr = jax.pmap(lambda x: x, devices=jax.local_devices())(local_arr_host)

    return local_arr

# <<<<<<<<<<<<<<<<<<<<<<

def pmap_devices_updated(pmapDevices):
    if collections.Counter(pmapDevices) == collections.Counter(myPmapDevices):
        return False
    return True


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


def set_pmap_devices(devices):
    devices = list(get_iterable(devices))
    global myPmapDevices
    global myDeviceCount
    global pmap_for_my_devices
    myPmapDevices = devices
    myDeviceCount = len(myPmapDevices)
    pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)
    myDevice = myPmapDevices[0]


def device_count():
    return len(myPmapDevices)


def devices():
    return myPmapDevices
