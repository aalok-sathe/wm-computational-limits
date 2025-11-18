from pathlib import Path
import yaml
import workingmem.model


# find model with best and worst val_acc
def best_worst(models_dir: Path, top_n: int = 1, verbose=False):
    # best = None
    # worst = None
    # best_acc = 0
    # worst_acc = 1

    models = []

    for model in models_dir.iterdir():
        try:
            with open(model / "history.yaml", "r") as f:
                val_acc = yaml.load(f, Loader=yaml.FullLoader)[-1]["eval_acc"]
                models.append((model, val_acc))
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     best = model
            # if val_acc < worst_acc:
            #     worst_acc = val_acc
            #     worst = model
        except TypeError:
            print(f"Skipping {model} due to TypeError")
            continue

    # sort models by val_acc
    models.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"Best models: {models[:top_n]}")
        print(f"Worst models: {models[-top_n:]}")
    for i in range(top_n):
        if i >= len(models):
            break
        yield workingmem.model.ModelConfig(models[i][0])
    for i in range(top_n):
        if i >= len(models):
            break
        yield workingmem.model.ModelConfig(models[-1 - i][0])
    # return map(workingmem.model.ModelConfig, [best, worst])
