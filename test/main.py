import sys
sys.path.append('/userhome/project/DDPNAS_V2')
import numpy as np
import tqdm
from distribution import Category_DDPNAS
from test_function import SumCategoryTestFunction


def get_optimizer(name, category):
    if name == 'DDPNAS':
        return Category_DDPNAS.CategoricalDDPNAS(category, 3)
    else:
        raise NotImplementedError


category = [8]*14
test_function = SumCategoryTestFunction(category)
optimizer_name = 'DDPNAS'

# distribution_optimizer = Category_DDPNAS.CategoricalDDPNAS(category, 3)
distribution_optimizer = get_optimizer(optimizer_name, category)
runing_times = 1
runing_epochs = 400
record = {
    'objective': np.zeros([runing_times, runing_epochs]) - 1,
    'l2_distance': np.zeros([runing_times, runing_epochs]) -1,
}
for i in tqdm.tqdm(range(runing_times)):
    for j in range(runing_epochs):
        if hasattr(distribution_optimizer, 'training_finish'):
            if distribution_optimizer.training_finish:
                break
        sample = distribution_optimizer.sampling_index()
        objective = test_function.objective_function(sample)
        distribution_optimizer.record_information(sample, objective)
        distribution_optimizer.update()
        current_best = np.argmax(distribution_optimizer.p_model.theta, axis=1)
        distance = test_function.l2_distance(current_best)
        record['objective'][i, j] = objective
        record['l2_distance'][i, j] = distance
    print(distribution_optimizer.p_model.theta)
    distribution_optimizer = get_optimizer(optimizer_name, category)
mean_obj = np.mean(record['objective'], axis=0)
mean_distance = np.mean(record['l2_distance'], axis=0)
print(mean_distance)
pass
