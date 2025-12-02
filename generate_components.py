from kfp.components import create_component_from_func
from src.pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

create_component_from_func(extract_data, output_component_file="components/extract_data.yaml")
create_component_from_func(preprocess_data, output_component_file="components/preprocess.yaml")
create_component_from_func(train_model, output_component_file="components/train.yaml")
create_component_from_func(evaluate_model, output_component_file="components/evaluate.yaml")
