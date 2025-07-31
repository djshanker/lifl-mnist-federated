
import csv
import matplotlib.pyplot as plt

def record_metrics(round_num, losses, accs):
    with open(f'metrics_round_{round_num}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Client', 'Loss', 'Accuracy'])
        for i, (loss, acc) in enumerate(zip(losses, accs)):
            writer.writerow([i, loss, acc])

def plot_results(metrics_dict):
    rounds = sorted(metrics_dict.keys())
    avg_acc = [sum(accs)/len(accs) for _, accs in metrics_dict.items()]
    plt.plot(rounds, avg_acc, marker='o')
    plt.title('Average Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('results.png')
