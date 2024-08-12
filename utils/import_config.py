import yaml

def config_parser(config_file):
    all_params = yaml.load(open(config_file, 'r'), yaml.FullLoader)

    if 'watermarker' in all_params:
        watermarker_params = all_params['watermarker']
    if 'sampler' in all_params:
        sampler_params = all_params['sampler']
    if 'model' in all_params:
        model_params = all_params['model']
    
    return watermarker_params, sampler_params, model_params