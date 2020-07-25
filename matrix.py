import numpy as np

# mat = np.array([[[-1,-2],[-3,-4],[-3,-4]],[[1,1],[2,-5],[2,-5]],[[1,1],[2,-6],[2,-6]]])
def mat_min(mat):
    mat_flat = mat.flatten()
    result = []
    step = len(mat[0].flatten())
    for i in range(step):
        result.append(min(mat_flat[i::step]))
    result = np.reshape(np.array(result),mat[0].shape)
    return result

def mat_absmin(mat):
    abs_index = mat_argmin(abs(mat))
    result = np.zeros(abs_index.shape)
    for row in range(abs_index.shape[0]):
        for col in range(abs_index.shape[1]):
            ind = abs_index[row][col]
            result[row][col] = mat[ind][row][col]
    return result
    
def mat_argmin(mat):
    mat_flat = mat.flatten()
    result_index = []
    step = len(mat[0].flatten())
    for i in range(step):
        result_index.append(np.argmin(mat_flat[i::step]))
    result_index = np.reshape(np.array(result_index),mat[0].shape)
    return result_index

if __name__ == '__main__':
    mat = np.array([[[0,0,0],[1,1,1]],[[0,0,0],[1,1,1]]])
    
    