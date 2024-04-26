import dace
import numpy as np
import cupy as cp
from dace.dtypes import StorageType, ScheduleType
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import unnecessary_host_gpu_transer_elimination

af = unnecessary_host_gpu_transer_elimination.UnnecessaryHostGPUTransferElimination()
remove_unnecessary_cpu_gpu_copies = ppl.Pipeline([af])

size = int(1e4)
N = dace.symbol('N')

@dace.program
def kernelInitAOnGPU(dataA : dace.float32[N] @ StorageType.GPU_Global):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataA[i] = 2.0

@dace.program
def kernelInitAOnCPU(dataA : dace.float32[N] @ StorageType.Default):
    for i in dace.map[0:N] @ ScheduleType.CPU_Multicore :
        dataA[i] = 2.0

@dace.program
def kernelInitBOnGPU(dataB : dace.float32[N] @ StorageType.GPU_Global):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataB[i] = 0.65

@dace.program
def kernelInitBOnCPU(dataB : dace.float32[N] @ StorageType.Default):
    for i in dace.map[0:N] @ ScheduleType.CPU_Multicore:
        dataB[i] = 0.65

@dace.program
def copyToGPU(data : dace.float32[N] @ StorageType.Default):
    # Does not compile if: gpu_data : dace.float32[N] @ StorageType.GPU_Global = cp.array(data)
    gpu_data : dace.float32[N] @ StorageType.GPU_Global = cp.zeros((N,), dace.float32)
    gpu_data[0:N] = data[0:N]
    return gpu_data

@dace.program
def kernelOverwriteBAddBToAOnGPU(
    dataA : dace.float32[N] @ StorageType.GPU_Global,
    dataB : dace.float32[N] @ StorageType.GPU_Global
    ):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataB[i] = 3.0
        dataA[i] += 0.5 * dataB[i]

@dace.program
def kernelOverwriteBAddBToAOnGPUNoWCR(
    dataA : dace.float32[N] @ StorageType.GPU_Global,
    dataB : dace.float32[N] @ StorageType.GPU_Global
    ):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataB[i] = 3.0
        dataA[i] = 0.5 * dataB[i] + dataA[i]

@dace.program
def kernelAddBToAOnGPU(
    dataA : dace.float32[N] @ StorageType.GPU_Global,
    dataB : dace.float32[N] @ StorageType.GPU_Global
    ):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataA[i] += 0.5 * dataB[i]

@dace.program
def kernelAddBToAOnGPUNoWCR(
    dataA : dace.float32[N] @ StorageType.GPU_Global,
    dataB : dace.float32[N] @ StorageType.GPU_Global
    ):
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        dataA[i] = 0.5 * dataB[i] + dataA[i]

@dace.program
def kernelAddBToAOnCPU(
    dataA : dace.float32[N] @ StorageType.Default,
    dataB : dace.float32[N] @ StorageType.Default
    ):
    for i in dace.map[0:N] @ ScheduleType.CPU_Multicore:
        dataA[i] += 0.5 * dataB[i]

@dace.program
def kernelOverwriteBAddBToAOnCPU(
    dataA : dace.float32[N] @ StorageType.Default,
    dataB : dace.float32[N] @ StorageType.Default
    ):
    for i in dace.map[0:N] @ ScheduleType.CPU_Multicore:
        dataB[i] = 3.0
        dataA[i] += 0.5 * dataB[i]


@dace.program
def kernelCoreOnGPU(dataA : dace.float32[N] @ StorageType.Default,
                  dataB : dace.float32[N] @ StorageType.Default):
    gpu_dataA = copyToGPU(data = dataA)
    gpu_dataB = copyToGPU(data = dataB)
    kernelInitAOnGPU(dataA = gpu_dataA)
    kernelInitBOnGPU(dataB = gpu_dataB)
    kernelOverwriteBAddBToAOnGPU(dataA = gpu_dataA, dataB = gpu_dataB)
    dataA[0:N] = gpu_dataA[0:N]

@dace.program
def kernelCoreOnCPU(dataA : dace.float32[N] @ StorageType.Default,
                  dataB : dace.float32[N] @ StorageType.Default):
    kernelInitAOnCPU(dataA = dataA)
    kernelInitBOnCPU(dataB = dataB)
    kernelOverwriteBAddBToAOnCPU(dataA = dataA, dataB = dataB)

@dace.program
def kernelLifetimesOnGPU(dataA : dace.float32[N] @ StorageType.Default,
                       dataB : dace.float32[N] @ StorageType.Default):
    gpu_dataA = copyToGPU(data = dataA)
    gpu_dataB = copyToGPU(data = dataB)
    kernelInitAOnGPU(dataA = gpu_dataA)
    kernelInitBOnGPU(dataB = gpu_dataB)
    kernelOverwriteBAddBToAOnGPU(dataA = gpu_dataA, dataB = gpu_dataB)
    dataA[0:N] = gpu_dataA[0:N]
    dataB[0:N] = gpu_dataB[0:N]
    kernelAddBToAOnCPU(dataA = dataA, dataB = dataB)
    gpu_dataA[0:N] = dataA[0:N]
    gpu_dataB[0:N] = dataB[0:N]
    kernelAddBToAOnGPU(dataA = gpu_dataA, dataB = gpu_dataB)
    dataA[0:N] = gpu_dataA[0:N]

@dace.program
def kernelLifetimesOnGPUNoWCR(dataA : dace.float32[N] @ StorageType.Default,
                       dataB : dace.float32[N] @ StorageType.Default):
    gpu_dataA = copyToGPU(data = dataA)
    gpu_dataB = copyToGPU(data = dataB)
    kernelInitAOnGPU(dataA = gpu_dataA)
    kernelInitBOnGPU(dataB = gpu_dataB)
    kernelOverwriteBAddBToAOnGPUNoWCR(dataA = gpu_dataA, dataB = gpu_dataB)
    dataA[0:N] = gpu_dataA[0:N]
    dataB[0:N] = gpu_dataB[0:N]
    kernelAddBToAOnCPU(dataA = dataA, dataB = dataB)
    gpu_dataA[0:N] = dataA[0:N]
    gpu_dataB[0:N] = dataB[0:N]
    kernelAddBToAOnGPUNoWCR(dataA = gpu_dataA, dataB = gpu_dataB)
    dataA[0:N] = gpu_dataA[0:N]

@dace.program
def kernelLifetimesOnCPU(dataA : dace.float32[N] @ StorageType.Default,
                          dataB : dace.float32[N] @ StorageType.Default):
    kernelInitAOnCPU(dataA = dataA)
    kernelInitBOnCPU(dataB = dataB)
    kernelOverwriteBAddBToAOnCPU(dataA = dataA, dataB = dataB)
    kernelAddBToAOnCPU(dataA = dataA, dataB = dataB)
    kernelAddBToAOnCPU(dataA = dataA, dataB = dataB)

def testCoreOnGPU():
    dataA = np.zeros(size, dtype=np.float32)
    dataB = np.zeros(size, dtype=np.float32)
    reference = np.full(size, 3.5, dtype=np.float32)
    sdfg = kernelCoreOnGPU.to_sdfg(dataA = dataA, dataB = dataB, N=size)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testCoreOnGPU:", results)
    sdfg(dataA = dataA, dataB = dataB, N=size)
    assert(np.allclose(dataA, reference))

def testCoreOnCPU():
    dataA = np.zeros(size, dtype=np.float32)
    dataB = np.zeros(size, dtype=np.float32)
    reference = np.full(size, 3.5, dtype=np.float32)
    sdfg = kernelCoreOnCPU.to_sdfg(dataA = dataA, dataB = dataB)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testCoreOnCPU:", results)
    sdfg(dataA = dataA, dataB = dataB, N = size)
    assert(np.allclose(dataA, reference))

def testLifetimesOnGPU():
    dataA = np.zeros(size, dtype=np.float32)
    dataB = np.zeros(size, dtype=np.float32)
    reference = np.full(size, 6.5, dtype=np.float32)
    sdfg =  kernelLifetimesOnGPU.to_sdfg(dataA=dataA, dataB=dataB, N = size)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testLifetimesOnGPU:", results)
    sdfg(dataA = dataA, dataB = dataB, N=size)
    assert(np.allclose(dataA, reference))

def testLifetimesOnGPUNoWCR():
    dataA = np.zeros(size, dtype=np.float32)
    dataB = np.zeros(size, dtype=np.float32)
    reference = np.full(size, 6.5, dtype=np.float32)
    sdfg = kernelLifetimesOnGPUNoWCR.to_sdfg(dataA=dataA, dataB=dataB, N = size)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testLifetimesOnGPUNoWCR:", results)
    sdfg(dataA = dataA, dataB = dataB, N=size)
    assert(np.allclose(dataA, reference))

def testLifetimesOnCPU():
    dataA = np.zeros(size, dtype=np.float32)
    dataB = np.zeros(size, dtype=np.float32)
    reference = np.full(size, 6.5, dtype=np.float32)
    sdfg = kernelLifetimesOnCPU.to_sdfg(dataA=dataA, dataB=dataB, N = size)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testLifetimesOnCPU:", results)
    sdfg(dataA = dataA, dataB = dataB, N=size)
    assert(np.allclose(dataA, reference))

def testNestedSDFG():
    ######################################
    # Create SDFG and sub-SDFG
    N = dace.symbol('N')
    sdfg = dace.SDFG('nested_main')
    sub_sdfg = dace.SDFG('nested_sub')

    # Create the arrays
    sdfg.add_array('A', [N], dace.float32, transient=False)
    sdfg.add_array('gpuA', [N], dace.float32, StorageType.GPU_Global, transient=True)
    sdfg.add_array('B', [N], dace.float32, transient=False)
    sdfg.add_array('gpuB', [N], dace.float32,  StorageType.GPU_Global, transient=True)

    ######################################
    # Create a copy from GPU to CPU for both arrays A and B

    state = sdfg.add_state('s0')
    an1 = state.add_read('A')
    an2 = state.add_write('gpuA')
    an1c1 = an1.add_out_connector('c1')
    an2c1 = an2.add_in_connector('c2')
    mem = dace.Memlet(expr="A")
    state.add_edge(an1, 'c1', an2, 'c2', memlet=mem)

    b_an1 = state.add_read('B')
    b_an2 = state.add_write('gpuB')
    b_an1c1 = b_an1.add_out_connector('c6')
    b_an2c1 = b_an2.add_in_connector('c7')
    b_mem = dace.Memlet(expr='B')
    state.add_edge(b_an1, 'c6', b_an2, 'c7', memlet=b_mem)
    sdfg.validate()

    ######################################
    # Create sub-SDFG input and output

    s_state0 = sub_sdfg.add_state('t0')
    sub_sdfg.add_array('s_gpuA', [N],
                        dace.float32, StorageType.GPU_Global)
    sub_sdfg.add_array('s_gpuB', [N],
                        dace.float32, StorageType.GPU_Global)

    nsdfg = state.add_nested_sdfg(sub_sdfg, sdfg, {'s_gpuA', 's_gpuB'}, {'s_gpuA', 's_gpuB'})

    ######################################
    # Move arrays from outside to the subsdfg

    mem2 = dace.Memlet(expr="gpuA")
    an2c2 = an2.add_out_connector('c3')
    state.add_edge(an2, 'c3', nsdfg, 's_gpuA', memlet=mem2)

    s_an1 = s_state0.add_read('s_gpuA')

    b_mem2 = dace.Memlet(expr="gpuB")
    b_an2c2 = b_an2.add_out_connector('c8')
    state.add_edge(b_an2, 'c8', nsdfg, 's_gpuB', memlet=b_mem2)

    b_s_an1 = s_state0.add_read('s_gpuB')

    me, mx = s_state0.add_map('mymap', dict(k='0:N'))

    ######################################
    # Task let code for vector addition A = A + B + 1

    s_t1 = s_state0.add_tasklet('add', {'a', 'b'}, {'c'}, 'c = a + b + 1')
    s_an2 = s_state0.add_write('s_gpuA')

    s_state0.add_memlet_path(s_an1, me, s_t1,
                        dst_conn='a',
                        memlet=dace.Memlet(data='s_gpuA', subset='k'))
    s_state0.add_memlet_path(b_s_an1, me, s_t1,
                        dst_conn='b',
                        memlet=dace.Memlet(data='s_gpuB', subset='k'))
    s_state0.add_memlet_path(s_t1, mx, s_an2,
                        src_conn='c',
                        memlet=dace.Memlet(data='s_gpuA', subset='k'))

    ######################################

    # Exit from SDFG, write the results to gpuA and gpuB
    # And then back to CPU A and B
    an3 = state.add_write('gpuA')
    an3c1 = an3.add_in_connector('c3')
    an3c2 = an3.add_out_connector('c4')
    state.add_edge(nsdfg, 's_gpuA', an3, 'c3', memlet=dace.Memlet(expr='gpuA'))

    an4 = state.add_write('A')
    an4c1 = an4.add_in_connector('c5')
    state.add_edge(an3, 'c4', an4, 'c5', memlet=dace.Memlet(expr='gpuA'))

    b_an3 = state.add_write('gpuB')
    b_an3c1 = b_an3.add_in_connector('c9')
    b_an3c2 = b_an3.add_out_connector('c10')
    state.add_edge(nsdfg, 's_gpuB', b_an3, 'c9', memlet=dace.Memlet(expr='gpuB'))

    b_an4 = state.add_write('B')
    b_an4c1 = b_an4.add_in_connector('c11')
    state.add_edge(b_an3, 'c10', b_an4, 'c11', memlet=dace.Memlet(expr='gpuB'))


    ######################################

    # Call the SDFG
    dataA = np.full((size,), 2.0, np.float32)
    dataB = np.full((size,), 3.0, np.float32)
    reference = np.full((size,), 6.0, np.float32)
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testNestedSDFG", results)
    sdfg(A=dataA, B=dataB, N=size)
    assert(np.allclose(dataA, reference))

@dace.program
def kernelMatAdd(A : dace.float32[N, N] @ StorageType.Default,
                                 B : dace.float32[N, N] @ StorageType.Default,
                                 C : dace.float32[N, N] @ StorageType.Default,
                                 D : dace.float32[N, N] @ StorageType.Default):
    # Perform matrix multiplication using for loops
    D_tmp : dace.float32[N, N] @ StorageType.GPU_Global = cp.zeros((N,N), cp.float32)
    gpuA : dace.float32[N, N] @ StorageType.GPU_Global = cp.zeros((N,N), cp.float32)
    gpuB : dace.float32[N, N] @ StorageType.GPU_Global = cp.zeros((N,N), cp.float32)
    gpuC : dace.float32[N, N] @ StorageType.GPU_Global = cp.zeros((N,N), cp.float32)
    gpuD : dace.float32[N, N] @ StorageType.GPU_Global = cp.zeros((N,N), cp.float32)
    gpuA[0:N, 0:N] = A[0:N, 0:N]
    gpuB[0:N, 0:N] = B[0:N, 0:N]
    gpuC[0:N, 0:N] = C[0:N, 0:N]
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        for j in dace.map[0:N] @ ScheduleType.GPU_ThreadBlock:
                D_tmp[i, j] += gpuA[i, j] + gpuB[i, j]
    for i in dace.map[0:N] @ ScheduleType.GPU_Device:
        for j in dace.map[0:N] @ ScheduleType.GPU_ThreadBlock:
                gpuD[i, j] += D_tmp[i, j] + gpuC[i, j]
    # The transformation can't catch this, as gpuA starts with 0
    # then it is written to, although it is the same value as A
    D[0:N, 0:N] = gpuD[0:N, 0:N]
    A[0:N, 0:N] = gpuA[0:N, 0:N]
    B[0:N, 0:N] = gpuB[0:N, 0:N]
    C[0:N, 0:N] = gpuC[0:N, 0:N]


def testMatAdd():
    size = 32
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    C = np.random.rand(size, size).astype(np.float32)
    D = np.zeros((size, size), np.float32)
    D_ref = A + B + C
    # Compute the result using the matrix multiplication kernel
    sdfg = kernelMatAdd.to_sdfg(A=A, B=B, C=C, D=D, N=size)
    sdfg.save("A.sdfg")
    results = remove_unnecessary_cpu_gpu_copies.apply_pass(sdfg, {})
    print("testMatAdd:", results)
    sdfg.save("A_opt.sdfg")
    sdfg(A=A, B=B, C=C, D=D, N=size)
    assert(np.allclose(D, D_ref))

if __name__ == '__main__':
    testCoreOnGPU()
    testCoreOnCPU()
    testLifetimesOnGPU()
    testLifetimesOnGPUNoWCR()
    testLifetimesOnCPU()
    testNestedSDFG()
    testMatAdd()