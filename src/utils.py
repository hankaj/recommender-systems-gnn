import csv
from datetime import datetime
import os

def save_metrics_to_file(training_type, time, args, precision_list, recall_list, hits_list):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"metrics_{current_datetime}.csv"
    folder_path = 'results/' + training_type

    # Create the results folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'w', newline='') as csvfile:
        csvfile.write(f"# Arguments: {args}\n")
        csvfile.write(f"# Training time: {time}\n")
        fieldnames = ['Epoch', 'Precision', 'Recall', 'Hits']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write metrics data
        for epoch, (precision, recall, hits) in enumerate(zip(precision_list, recall_list, hits_list)):
            writer.writerow({'Epoch': epoch, 'Precision': precision, 'Recall': recall, 'Hits': hits})
        print(f"Metrics saved to {file_path}")
