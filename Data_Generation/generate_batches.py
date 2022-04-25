from numpy import savetxt
from PSO import PSO
from complexity_functions import ComplexityFunction
from data_generator import *

# TODO Refactor!!

objective_function = ComplexityFunction.N1
complexity_type = 'neighborhood_complexity_check'
target_measure = 'N1'
min_complex = 0
max_complex = 0.80
number_of_batches = 16
target_values = list(np.linspace(min_complex, max_complex, num=number_of_batches))


test_samples = []
main_dataset = []

for i in range(len(target_values)):
    complexity_value, param = PSO(complexity=target_values[i], complexity_function=objective_function)
    dataset = generate_data(125, param)
    for j in range(25):
        random_index = np.random.randint(len(dataset))
        test_samples.append(dataset[random_index])
        dataset = np.delete(dataset, random_index, 0)
    complexity_value = validate_complexity(dataset, (complexity_type, target_measure))
    savetxt(
        '../Data_Generation/data/{}{}.csv'.format(np.round(complexity_value, 5), (i % 10)),
        dataset, delimiter=',')
    save_plot(dataset, '../Data_Generation/data/{}{}.png'.format(np.round(complexity_value, 5), (i % 10)),
              complexity_value)
    main_dataset.append(dataset)

# combine the batches to make main_dataset
combined_dataset = np.array(main_dataset).reshape(1600, 3)
combined_complexity = validate_complexity(combined_dataset, (complexity_type, target_measure))
savetxt('../Data_Generation/data/combined_dataset_{}_complexity_{}.csv'.format(objective_function.__name__,
                                                                               np.round(combined_complexity, 5)),
        combined_dataset, delimiter=',')
save_plot(combined_dataset,
          '../Data_Generation/data/combined_dataset_{}_complexity_{}.png'.format(objective_function.__name__,
                                                                                 np.round(combined_complexity, 5)),
          combined_complexity)

# generate the test set
test_data = np.array(test_samples).reshape(400, 3)
test_complexity = validate_complexity(test_data, (complexity_type, target_measure))
savetxt(
    '../Data_Generation/data/test_dataset_{}_complexity_{}.csv'.format(objective_function.__name__,
                                                                       np.round(test_complexity, 5)),
    test_data, delimiter=',')
save_plot(test_data, '../Data_Generation/data/test_dataset_{}_complexity_{}.png'.format(objective_function.__name__,
                                                                                        np.round(test_complexity, 5)),
          test_complexity)
