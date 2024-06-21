import numpy as np
import matplotlib.pyplot as plt

# 定义函数来生成变换后的相似性度量值
def transform_similarity(similarity_matrix, hyper_params):
    # 解析参数
    scale_factor = hyper_params[0]
    denominator_offset = hyper_params[1]
    constant_offset = hyper_params[2]
    exponent_factor = hyper_params[3]
    
    # 对相似性度量矩阵进行变换
    transformed_matrix = scale_factor / (denominator_offset + np.exp(-(similarity_matrix - constant_offset) * exponent_factor))
    
    return transformed_matrix
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 生成自变量范围为0到1的相似性度量矩阵
similarity_matrix = np.linspace(0, 1, 100)  # 选择合适的x范围以展示sigmoid函数的形状

# 定义示例的超参数
hyper_params = [1, 1.0, 0, 1.0]

# 变换相似性度量
transformed_similarity = transform_similarity(similarity_matrix, hyper_params)
sigmoid_similarity = sigmoid(similarity_matrix)


def draw_0():
    # 绘制原始相似性度量和变换后的相似性度量曲线
    plt.figure(figsize=(10, 10))
    plt.plot(similarity_matrix,transformed_similarity, label=str(hyper_params))
    hyper_params_1 = [1, 1.0, 0, 1.0]
    hyper_params_2 = [1, 1.0, 0.75, 10.0]
    plt.plot(similarity_matrix, transform_similarity(similarity_matrix, hyper_params_1), label=str(hyper_params_1))
    plt.plot(similarity_matrix, transform_similarity(similarity_matrix, hyper_params_2), label=str(hyper_params_2))
    plt.title('Transformation of Similarity Measure')
    plt.xlabel('Similarity Measure')
    plt.ylabel('Transformed Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def draw_1():
    # 计算sigmoid函数的值
    sigmoid_values = sigmoid(similarity_matrix)

    # 绘制sigmoid函数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(similarity_matrix, sigmoid_values, label='Sigmoid Function')
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
def draw_2():
    plt.figure(figsize=(10,10  ))
    hyper_param_range = np.arange(0, 1, 0.05)
    for i in hyper_param_range:
        hyper_params = [1, 1, i, 10]
        transformed_similarity = transform_similarity(similarity_matrix, hyper_params)
        plt.plot(similarity_matrix, transformed_similarity, label=str(hyper_params))
    plt.title('Transformation of Similarity Measure')
    plt.xlabel('Similarity Measure')
    plt.ylabel('Transformed Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def draw_3():
    plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        hyper_params = [1, 1.0, 0, i]
        transformed_similarity = transform_similarity(similarity_matrix, hyper_params)
        plt.plot(similarity_matrix, transformed_similarity, label=str(hyper_params))
    plt.title('Transformation of Similarity Measure')
    plt.xlabel('Similarity Measure')
    plt.ylabel('Transformed Value')
    plt.legend()
    plt.grid(True)
    plt.show()
draw_0()
# draw_2()
#draw_3()