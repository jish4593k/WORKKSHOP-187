import os
import dropbox
from dropbox.files import WriteMode
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedTransferData:
    def __init__(self, access_token):
        self.access_token = access_token
        self.transfer_history = {'dates': [], 'file_counts': []}
        self.file_extensions = []

    def upload_files_to_dropbox(self, source_folder, destination_folder):
        dbx = dropbox.Dropbox(self.access_token)

        for root, dirs, files in os.walk(source_folder):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, source_folder)
                dropbox_path = os.path.join(destination_folder, filename)

                with open(local_path, 'rb') as file:
                    dbx.files_upload(file.read(), dropbox_path, mode=WriteMode('overwrite'))

                # Extract file extension for data mining
                _, file_extension = os.path.splitext(filename)
                self.file_extensions.append(file_extension.lower())

        # Record the transfer details for regression analysis
        self.record_transfer(datetime.now(), len(files))

        print("Files have been moved to Dropbox!")

    def record_transfer(self, date, file_count):
        self.transfer_history['dates'].append(date)
        self.transfer_history['file_counts'].append(file_count)

    def perform_regression_analysis(self):
        # Perform linear regression on the number of files transferred over time
        x_values = [(date - self.transfer_history['dates'][0]).days for date in self.transfer_history['dates']]
        y_values = self.transfer_history['file_counts']

        regression_model = LinearRegression()
        x_values_reshaped = [[x] for x in x_values]
        regression_model.fit(x_values_reshaped, y_values)
        predicted_file_counts = regression_model.predict(x_values_reshaped)

        # Plot the trend
        self.plot_regression_trend(x_values, y_values, predicted_file_counts)

    def plot_regression_trend(self, x_values, actual_values, predicted_values):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x_values, y=actual_values, label='Actual File Counts')
        sns.lineplot(x=x_values, y=predicted_values, label='Predicted File Counts', linestyle='dashed')
        plt.title('Regression Analysis of File Transfer History')
        plt.xlabel('Days since the first transfer')
        plt.ylabel('Number of Files Transferred')
        plt.legend()
        plt.show()

    def perform_data_mining(self):
        # Simple data mining: Analyze file extensions using CountVectorizer and KMeans clustering
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.file_extensions)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)

        # Visualize clustering results
        self.plot_cluster_analysis(X, kmeans.labels_)

    def plot_cluster_analysis(self, X, labels):
        plt.figure(figsize=(8, 6))
        plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.title('File Extension Clustering Analysis')
        plt.xlabel('Count of Extensions')
        plt.ylabel('Count of Different Extensions')
        plt.show()

def main():
    access_token = 'pkkvbulg9mYAAAAAAAAAAX7U8NNsSfzoIcmTLLVOJhvaJ0iVvQIJa22EiUST5MFA'
    advanced_transfer_data = AdvancedTransferData(access_token)

    # GUI for selecting source and destination folders
    root = tk.Tk()
    root.withdraw()
    source_folder = filedialog.askdirectory(title="Select Source Folder")
    destination_folder = filedialog.askdirectory(title="Select Destination Folder")

    advanced_transfer_data.upload_files_to_dropbox(source_folder, destination_folder)
    advanced_transfer_data.perform_regression_analysis()
    advanced_transfer_data.perform_data_mining()

if __name__ == "__main__":
    main()
