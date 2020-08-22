import yaml

file = '../envs/configs/base_sim.yaml'
with open(file) as f:
    config = yaml.load(f, yaml.FullLoader)

for k in config:
    print(type(config[k]))
    if type(config[k]) == list:
        print(type(config[k][0]))
    print(config[k])
