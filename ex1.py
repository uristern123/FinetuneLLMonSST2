import os
import torch
import time

from matplotlib import pyplot as plt

os.environ['HF_DATASETS_CACHE'] = './'
os.environ['HF_METRICS_CACHE'] = './'
os.environ['HF_MODULES_CACHE'] = './'
os.environ['HF_DATASETS_DOWNLOADED_EVALUATE_PATH'] = './'
os.environ['TRANSFORMERS_CACHE'] = './'

from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, set_seed, DataCollatorWithPadding
from dataclasses import dataclass, field
import evaluate
import numpy as np
import argparse
import accelerate



# Step 1: Load arguments from user
parser = argparse.ArgumentParser(description='finetuning LLMs on SST2')
parser.add_argument('--num_seeds', type=int, default=3, help='number of seeds to use')
parser.add_argument('--train_samples_amount', type=int, default=-1, help='amount of train data to use (-1 means all data)')
parser.add_argument('--val_samples_amount', type=int, default=-1, help='amount of validation data to use (-1 means all data)')
parser.add_argument('--test_samples_amount', type=int, default=-1, help='amount of test data to use (-1 means all data)')
args = parser.parse_args()

# Step 2: Load dataset
dataset = load_dataset("sst2")
if args.train_samples_amount != -1:
    dataset["train"] = dataset["train"].select(range(args.train_samples_amount))
if args.val_samples_amount != -1:
    dataset["validation"] = dataset["validation"].select(range(args.val_samples_amount))
if args.test_samples_amount != -1:
    dataset["test"] = dataset["test"].select(range(args.test_samples_amount))

# Step 3: Define evaluation metric (accuracy)
metric = evaluate.load("accuracy")
def accuracy(pred_and_labels):
    logits, labels = pred_and_labels
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model_names = ['roberta-base', 'bert-base-uncased', 'google/electra-base-generator']
total_val_acc = []
mean_val_accs = []
all_models = []
start = time.time()
for model_name in model_names:
    all_models.append([])
    val_acc = np.empty(args.num_seeds)
    for seed in range(args.num_seeds):
        set_seed(seed)
        training_args = TrainingArguments(
            output_dir="./"+model_name+str(seed)+"results",
        )


        # Step 4: Load model and tokenizer
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        all_models[-1].append(AutoModelForSequenceClassification.from_pretrained(model_name, config=config).cuda())
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

        def tokenize_function(example):
            return tokenizer(example["sentence"], max_length=512,truncation=True)  # 512 because for all three models max_length = 512

        # Step 5: tokenize dataset. Notice that we use dynamic padding, so padding will only be done per batch.
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Step 6: Train model
        trainer = Trainer(
            model=all_models[-1][seed],
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=accuracy,
            tokenizer = tokenizer,
            data_collator = data_collator
        )

        trainer.train()
        # Step 7: Evaluate model
        all_models[-1][seed].eval()
        eval_result = trainer.evaluate(tokenized_dataset["validation"])
        val_acc[seed] = eval_result["eval_accuracy"]
        print("Validation Accuracy on seed %d: %d", seed, eval_result["eval_accuracy"])

    total_val_acc.append(val_acc)
    mean_val_acc = np.mean(val_acc)
    std_val_acc = np.std(val_acc)
    mean_val_accs.append(mean_val_acc)
    with open("./res.txt", 'a') as f:
        f.write(model_name + "," + str(np.mean(val_acc)) + " +- "+ str(np.std(val_acc)) + '\n')
    f.close()
end = time.time()
with open("./res.txt", 'a') as f:
    f.write("----" + '\n')
    f.write("train time," + str(end - start) + '\n')
f.close()
#Step 8: choose best model and best seed
best_seed_per_model = np.max(np.asarray(total_val_acc), axis = 1)
best_model = np.argmax(best_seed_per_model)
print("best model is: %s" % model_names[best_model])
best_seed = np.argmax(total_val_acc[np.argmax(best_seed_per_model)])
print("best model is: %d" % best_seed)

#Step 9:#since we don't want to pad the test examples, we'll provide them one by one to the model, saving predictions in a dedicated file
tokenizer = AutoTokenizer.from_pretrained(model_names[best_model])
dataset["test"] = dataset["test"].remove_columns("label")
model = all_models[best_model][best_seed]
model.eval()
start = time.time()
with open("./"+model_names[best_model]+str(best_seed)+"results/predictions.txt", 'a') as f:
    for sample in dataset["test"]:
        encoded_input = tokenizer(sample['sentence'], return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())

        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()

        # Convert predicted class index back to label
        label_list = model.config.label2id
        predicted_label = list(label_list.keys())[list(label_list.values()).index(predicted_class_idx)]
        f.write(sample['sentence']+"###"+str(predicted_label)+'\n')
end = time.time()
with open("./res.txt", 'a') as f:
    f.write("prediction time,"+str(end-start) + '\n')
f.close()

#creating a loss graph of the first seed of the different models. I have done this manually here with the output loss of the models, but it is probably best to use wandb for this.
roberta_training_log = [
    {'loss': 0.5598, 'learning_rate': 4.9010175396919665e-05, 'epoch': 0.06},
    {'loss': 0.4817, 'learning_rate': 4.8020350793839334e-05, 'epoch': 0.12},
    {'loss': 0.4384, 'learning_rate': 4.7030526190759e-05, 'epoch': 0.18},
    {'loss': 0.4413, 'learning_rate': 4.6040701587678666e-05, 'epoch': 0.24},
    {'loss': 0.4174, 'learning_rate': 4.5050876984598335e-05, 'epoch': 0.3},
    {'loss': 0.4486, 'learning_rate': 4.4061052381518e-05, 'epoch': 0.36},
    {'loss': 0.425, 'learning_rate': 4.307122777843766e-05, 'epoch': 0.42},
    {'loss': 0.4068, 'learning_rate': 4.208140317535733e-05, 'epoch': 0.48},
    {'loss': 0.4008, 'learning_rate': 4.109157857227699e-05, 'epoch': 0.53},
    {'loss': 0.3881, 'learning_rate': 4.010175396919666e-05, 'epoch': 0.59},
    {'loss': 0.3947, 'learning_rate': 3.9111929366116324e-05, 'epoch': 0.65},
    {'loss': 0.3731, 'learning_rate': 3.812210476303599e-05, 'epoch': 0.71},
    {'loss': 0.4306, 'learning_rate': 3.7132280159955656e-05, 'epoch': 0.77},
    {'loss': 0.4012, 'learning_rate': 3.614245555687532e-05, 'epoch': 0.83},
    {'loss': 0.4217, 'learning_rate': 3.515263095379499e-05, 'epoch': 0.89},
    {'loss': 0.3909, 'learning_rate': 3.4162806350714657e-05, 'epoch': 0.95},
    {'loss': 0.507, 'learning_rate': 3.317298174763432e-05, 'epoch': 1.01},
    {'loss': 0.4126, 'learning_rate': 3.218315714455399e-05, 'epoch': 1.07},
    {'loss': 0.4753, 'learning_rate': 3.119333254147365e-05, 'epoch': 1.13},
    {'loss': 0.3964, 'learning_rate': 3.0203507938393317e-05, 'epoch': 1.19},
    {'loss': 0.3856, 'learning_rate': 2.9213683335312986e-05, 'epoch': 1.25},
    {'loss': 0.5514, 'learning_rate': 2.822385873223265e-05, 'epoch': 1.31},
    {'loss': 0.489, 'learning_rate': 2.7234034129152314e-05, 'epoch': 1.37},
    {'loss': 0.4315, 'learning_rate': 2.6244209526071984e-05, 'epoch': 1.43},
    {'loss': 0.4224, 'learning_rate': 2.5254384922991646e-05, 'epoch': 1.48},
    {'loss': 0.3965, 'learning_rate': 2.4264560319911315e-05, 'epoch': 1.54},
    {'loss': 0.3822, 'learning_rate': 2.3274735716830978e-05, 'epoch': 1.6},
    {'loss': 0.3867, 'learning_rate': 2.2284911113750644e-05, 'epoch': 1.66},
    {'loss': 0.3555, 'learning_rate': 2.129508651067031e-05, 'epoch': 1.72},
    {'loss': 0.3613, 'learning_rate': 2.0305261907589976e-05, 'epoch': 1.78},
    {'loss': 0.3501, 'learning_rate': 1.931543730450964e-05, 'epoch': 1.84},
    {'loss': 0.3326, 'learning_rate': 1.8325612701429307e-05, 'epoch': 1.9},
    {'loss': 0.3086, 'learning_rate': 1.7335788098348973e-05, 'epoch': 1.96},
    {'loss': 0.2826, 'learning_rate': 1.634596349526864e-05, 'epoch': 2.02},
    {'loss': 0.2944, 'learning_rate': 1.5356138892188305e-05, 'epoch': 2.08},
    {'loss': 0.2746, 'learning_rate': 1.4366314289107971e-05, 'epoch': 2.14},
    {'loss': 0.286, 'learning_rate': 1.3376489686027638e-05, 'epoch': 2.2},
    {'loss': 0.2593, 'learning_rate': 1.2386665082947303e-05, 'epoch': 2.26},
    {'loss': 0.2792, 'learning_rate': 1.1396840479866969e-05, 'epoch': 2.32},
    {'loss': 0.2362, 'learning_rate': 1.0407015876786634e-05, 'epoch': 2.38},
    {'loss': 0.2293, 'learning_rate': 9.417191273706299e-06, 'epoch': 2.43},
    {'loss': 0.2493, 'learning_rate': 8.427366670625965e-06, 'epoch': 2.49},
    {'loss': 0.2314, 'learning_rate': 7.437542067545632e-06, 'epoch': 2.55},
    {'loss': 0.252, 'learning_rate': 6.447717464465297e-06, 'epoch': 2.61},
    {'loss': 0.2522, 'learning_rate': 5.457892861384962e-06, 'epoch': 2.67},
    {'loss': 0.2356, 'learning_rate': 4.468068258304629e-06, 'epoch': 2.73},
    {'loss': 0.2252, 'learning_rate': 3.478243655224295e-06, 'epoch': 2.79},
    {'loss': 0.2214, 'learning_rate': 2.4884190521439603e-06, 'epoch': 2.85},
    {'loss': 0.237, 'learning_rate': 1.4985944490636262e-06, 'epoch': 2.91},
    {'loss': 0.2594, 'learning_rate': 5.087698459832917e-07, 'epoch': 2.97},
    {'loss': 0.3599065711238952, 'epoch': 3.0}
]
loss = []
epoch = []
for log in roberta_training_log:
    loss.append(log['loss'])
    epoch.append(log['epoch'])

plt.plot(epoch, loss, label = 'roberta-base')

Bert_training_logs = [
    {'loss': 0.4092, 'learning_rate': 4.9010175396919665e-05, 'epoch': 0.06},
    {'loss': 0.3772, 'learning_rate': 4.8020350793839334e-05, 'epoch': 0.12},
    {'loss': 0.349, 'learning_rate': 4.7030526190759e-05, 'epoch': 0.18},
    {'loss': 0.3401, 'learning_rate': 4.6040701587678666e-05, 'epoch': 0.24},
    {'loss': 0.303, 'learning_rate': 4.5050876984598335e-05, 'epoch': 0.3},
    {'loss': 0.303, 'learning_rate': 4.4061052381518e-05, 'epoch': 0.36},
    {'loss': 0.2964, 'learning_rate': 4.307122777843766e-05, 'epoch': 0.42},
    {'loss': 0.2818, 'learning_rate': 4.208140317535733e-05, 'epoch': 0.48},
    {'loss': 0.2895, 'learning_rate': 4.109157857227699e-05, 'epoch': 0.53},
    {'loss': 0.2797, 'learning_rate': 4.010175396919666e-05, 'epoch': 0.59},
    {'loss': 0.2539, 'learning_rate': 3.9111929366116324e-05, 'epoch': 0.65},
    {'loss': 0.2731, 'learning_rate': 3.812210476303599e-05, 'epoch': 0.71},
    {'loss': 0.2651, 'learning_rate': 3.7132280159955656e-05, 'epoch': 0.77},
    {'loss': 0.2653, 'learning_rate': 3.614245555687532e-05, 'epoch': 0.83},
    {'loss': 0.2509, 'learning_rate': 3.515263095379499e-05, 'epoch': 0.89},
    {'loss': 0.2461, 'learning_rate': 3.4162806350714657e-05, 'epoch': 0.95},
    {'loss': 0.2551, 'learning_rate': 3.317298174763432e-05, 'epoch': 1.01},
    {'loss': 0.1658, 'learning_rate': 3.218315714455399e-05, 'epoch': 1.07},
    {'loss': 0.1767, 'learning_rate': 3.119333254147365e-05, 'epoch': 1.13},
    {'loss': 0.1891, 'learning_rate': 3.0203507938393317e-05, 'epoch': 1.19},
    {'loss': 0.1816, 'learning_rate': 2.9213683335312986e-05, 'epoch': 1.25},
    {'loss': 0.1797, 'learning_rate': 2.822385873223265e-05, 'epoch': 1.31},
    {'loss': 0.1761, 'learning_rate': 2.7234034129152314e-05, 'epoch': 1.37},
    {'loss': 0.1738, 'learning_rate': 2.6244209526071984e-05, 'epoch': 1.43},
    {'loss': 0.1825, 'learning_rate': 2.5254384922991646e-05, 'epoch': 1.48},
    {'loss': 0.1676, 'learning_rate': 2.4264560319911315e-05, 'epoch': 1.54},
    {'loss': 0.1766, 'learning_rate': 2.3274735716830978e-05, 'epoch': 1.6},
    {'loss': 0.1634, 'learning_rate': 2.2284911113750644e-05, 'epoch': 1.66},
    {'loss': 0.1839, 'learning_rate': 2.129508651067031e-05, 'epoch': 1.72},
    {'loss': 0.1864, 'learning_rate': 2.0305261907589976e-05, 'epoch': 1.78},
    {'loss': 0.1789, 'learning_rate': 1.931543730450964e-05, 'epoch': 1.84},
    {'loss': 0.1575, 'learning_rate': 1.8325612701429307e-05, 'epoch': 1.9},
    {'loss': 0.1621, 'learning_rate': 1.7335788098348973e-05, 'epoch': 1.96},
    {'loss': 0.1236, 'learning_rate': 1.634596349526864e-05, 'epoch': 2.02},
    {'loss': 0.1003, 'learning_rate': 1.5356138892188305e-05, 'epoch': 2.08},
    {'loss': 0.105, 'learning_rate': 1.4366314289107971e-05, 'epoch': 2.14},
    {'loss': 0.0938, 'learning_rate': 1.3376489686027638e-05, 'epoch': 2.2},
    {'loss': 0.1011, 'learning_rate': 1.2386665082947303e-05, 'epoch': 2.26},
    {'loss': 0.1242, 'learning_rate': 1.1396840479866969e-05, 'epoch': 2.32},
    {'loss': 0.0938, 'learning_rate': 1.0407015876786634e-05, 'epoch': 2.38},
    {'loss': 0.0991, 'learning_rate': 9.417191273706299e-06, 'epoch': 2.43},
    {'loss': 0.101, 'learning_rate': 8.427366670625965e-06, 'epoch': 2.49},
    {'loss': 0.0903, 'learning_rate': 7.437542067545632e-06, 'epoch': 2.55},
    {'loss': 0.105, 'learning_rate': 6.447717464465297e-06, 'epoch': 2.61},
    {'loss': 0.1043, 'learning_rate': 5.457892861384962e-06, 'epoch': 2.67},
    {'loss': 0.0927, 'learning_rate': 4.468068258304629e-06, 'epoch': 2.73},
    {'loss': 0.0946, 'learning_rate': 3.478243655224295e-06, 'epoch': 2.79},
    {'loss': 0.094, 'learning_rate': 2.4884190521439603e-06, 'epoch': 2.85},
    {'loss': 0.0991, 'learning_rate': 1.4985944490636262e-06, 'epoch': 2.91},
    {'loss': 0.1102, 'learning_rate': 5.087698459832917e-07, 'epoch': 2.97},
    {'loss': 0.19054453953591455, 'epoch': 3.0}
]
loss = []
epoch = []
for log in Bert_training_logs:
    loss.append(log['loss'])
    epoch.append(log['epoch'])

plt.plot(epoch, loss, label = 'bert-base-uncased')
Electra_training_logs = [
    {'loss': 0.5439, 'learning_rate': 4.9010175396919665e-05, 'epoch': 0.06},
    {'loss': 0.4049, 'learning_rate': 4.8020350793839334e-05, 'epoch': 0.12},
    {'loss': 0.3476, 'learning_rate': 4.7030526190759e-05, 'epoch': 0.18},
    {'loss': 0.3237, 'learning_rate': 4.6040701587678666e-05, 'epoch': 0.24},
    {'loss': 0.3236, 'learning_rate': 4.5050876984598335e-05, 'epoch': 0.3},
    {'loss': 0.3292, 'learning_rate': 4.4061052381518e-05, 'epoch': 0.36},
    {'loss': 0.3105, 'learning_rate': 4.307122777843766e-05, 'epoch': 0.42},
    {'loss': 0.2974, 'learning_rate': 4.208140317535733e-05, 'epoch': 0.48},
    {'loss': 0.2844, 'learning_rate': 4.109157857227699e-05, 'epoch': 0.53},
    {'loss': 0.2817, 'learning_rate': 4.010175396919666e-05, 'epoch': 0.59},
    {'loss': 0.2643, 'learning_rate': 3.9111929366116324e-05, 'epoch': 0.65},
    {'loss': 0.256, 'learning_rate': 3.812210476303599e-05, 'epoch': 0.71},
    {'loss': 0.2725, 'learning_rate': 3.7132280159955656e-05, 'epoch': 0.77},
    {'loss': 0.2658, 'learning_rate': 3.614245555687532e-05, 'epoch': 0.83},
    {'loss': 0.2668, 'learning_rate': 3.515263095379499e-05, 'epoch': 0.89},
    {'loss': 0.2476, 'learning_rate': 3.4162806350714657e-05, 'epoch': 0.95},
    {'loss': 0.2497, 'learning_rate': 3.317298174763432e-05, 'epoch': 1.01},
    {'loss': 0.1873, 'learning_rate': 3.218315714455399e-05, 'epoch': 1.07},
    {'loss': 0.2045, 'learning_rate': 3.119333254147365e-05, 'epoch': 1.13},
    {'loss': 0.1829, 'learning_rate': 3.0203507938393317e-05, 'epoch': 1.19},
    {'loss': 0.1886, 'learning_rate': 2.9213683335312986e-05, 'epoch': 1.25},
    {'loss': 0.1975, 'learning_rate': 2.822385873223265e-05, 'epoch': 1.31},
    {'loss': 0.1794, 'learning_rate': 2.7234034129152314e-05, 'epoch': 1.37},
    {'loss': 0.1867, 'learning_rate': 2.6244209526071984e-05, 'epoch': 1.43},
    {'loss': 0.1935, 'learning_rate': 2.5254384922991646e-05, 'epoch': 1.48},
    {'loss': 0.1848, 'learning_rate': 2.4264560319911315e-05, 'epoch': 1.54},
    {'loss': 0.1943, 'learning_rate': 2.3274735716830978e-05, 'epoch': 1.6},
    {'loss': 0.1723, 'learning_rate': 2.2284911113750644e-05, 'epoch': 1.66},
    {'loss': 0.1958, 'learning_rate': 2.129508651067031e-05, 'epoch': 1.72},
    {'loss': 0.1927, 'learning_rate': 2.0305261907589976e-05, 'epoch': 1.78},
    {'loss': 0.1869, 'learning_rate': 1.931543730450964e-05, 'epoch': 1.84},
    {'loss': 0.1736, 'learning_rate': 1.8325612701429307e-05, 'epoch': 1.9},
    {'loss': 0.1888, 'learning_rate': 1.7335788098348973e-05, 'epoch': 1.96},
    {'loss': 0.1622, 'learning_rate': 1.634596349526864e-05, 'epoch': 2.02},
    {'loss': 0.116, 'learning_rate': 1.5356138892188305e-05, 'epoch': 2.08},
    {'loss': 0.1339, 'learning_rate': 1.4366314289107971e-05, 'epoch': 2.14},
    {'loss': 0.1253, 'learning_rate': 1.3376489686027638e-05, 'epoch': 2.2},
    {'loss': 0.1254, 'learning_rate': 1.2386665082947303e-05, 'epoch': 2.26},
    {'loss': 0.1278, 'learning_rate': 1.1396840479866969e-05, 'epoch': 2.32},
    {'loss': 0.1283, 'learning_rate': 1.0407015876786634e-05, 'epoch': 2.38},
    {'loss': 0.131, 'learning_rate': 9.417191273706299e-06, 'epoch': 2.43},
    {'loss': 0.1134, 'learning_rate': 8.427366670625965e-06, 'epoch': 2.49},
    {'loss': 0.125, 'learning_rate': 7.437542067545632e-06, 'epoch': 2.55},
    {'loss': 0.1328, 'learning_rate': 6.447717464465297e-06, 'epoch': 2.61},
    {'loss': 0.137, 'learning_rate': 5.457892861384962e-06, 'epoch': 2.67},
    {'loss': 0.1217, 'learning_rate': 4.468068258304629e-06, 'epoch': 2.73},
    {'loss': 0.1167, 'learning_rate': 3.478243655224295e-06, 'epoch': 2.79},
    {'loss': 0.1257, 'learning_rate': 2.4884190521439603e-06, 'epoch': 2.85},
    {'loss': 0.1339, 'learning_rate': 1.4985944490636262e-06, 'epoch': 2.91},
    {'loss': 0.1338, 'learning_rate': 5.087698459832917e-07, 'epoch': 2.97},
    {'loss': 0.20857037726512273, 'epoch': 3.0}
]
loss = []
epoch = []
for log in Electra_training_logs:
    loss.append(log['loss'])
    epoch.append(log['epoch'])

plt.plot(epoch, loss, label = 'electra-base-generator')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
