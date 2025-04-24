from neural import *

# print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# print("\n\nTraining XOR\n\n")
# xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

# xorn = NeuralNet(2, 8, 1)
# xorn.train(xor_training_data, iters=10000, print_interval=100)
# print(xorn.test_with_expected(xor_training_data))

print("<<<<<<<<<<<<<<Political Data>>>>>>>>>>>>>>>>>")

print("\n\nTraning Political\n")
political_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0])
]


political_net = NeuralNet(5, 5, 1)
political_net.train(political_training_data, iters=10000, print_interval=100)

new_input = [0.9, 0.8, 0.8, 0.3, 0.6]
prediction = political_net.evaluate(new_input)


print(f"Prediction for input {new_input}: {prediction}")