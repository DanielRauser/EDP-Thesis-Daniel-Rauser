import matplotlib.pyplot as plt
import mlflow

def log_fig(filename):
    fig = plt.gcf()
    mlflow.log_figure(fig, filename)
    plt.close(fig)