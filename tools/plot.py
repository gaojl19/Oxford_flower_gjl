def plot_loss_curve(curve, plot_prefix):
    
    import pandas as pd
    import seaborn as sns

    iteration = range(1, len(curve)+1)
    data = pd.DataFrame(curve, iteration)
    ax=sns.lineplot(data=data)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss Curve")
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + "loss_curve.png")
    
    fig.clf()


def plot_acc_curve(train_acc, valid_acc, plot_prefix):
    
    import pandas as pd
    import seaborn as sns

    assert len(train_acc)==len(valid_acc)
    iteration = range(1, len(train_acc)+1)
    df = {
        "train_acc": train_acc,
        "valid_acc": valid_acc
    }
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + "accuracy_curve.png")
    
    fig.clf()


if __name__ == "__main__":
    import json
    with open("./log/ConvMixer_lr0.001/log.json", "r") as f:
        json_file = json.load(f)
        
    plot_loss_curve(json_file["loss"], "./log/ConvMixer_lr0.001/")
    plot_acc_curve(json_file["train_acc"], json_file["valid_acc"],"./log/ConvMixer_lr0.001/")