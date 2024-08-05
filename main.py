import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv

# Load environment variables from .env fil
load_dotenv()

# Get the API key from the environment variable
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("API key is missing. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = API_KEY


def calculate_statistics(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode().values[0]  # mode() returns a Series
    std_dev = df[column].std()
    return mean, median, mode, std_dev


def plot_all_histograms(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)

    # Calculate the number of rows and columns for subplots
    num_rows = (num_columns + 2) // 3  # 3 columns per row

    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 4 * num_rows))
    axes = axes.flatten()  # Flatten to make iteration easier

    for i, column in enumerate(numeric_columns):
        axes[i].hist(df[column].dropna(), bins=30, edgecolor='k')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Remove any empty subplots if the number of columns is not a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    plt.ioff()  # Turn off interactive mode


def generate_insights(stats_summary):
    # Dummy function for generating insights
    print(f"Statistics Summary: {stats_summary}")


def ask_llm_question(df, question):
    # Convert DataFrame to a string representation
    data_description = df.describe(include='all').to_string()
    prompt = f"Based on the following dataset description:\n{data_description}\n\nAnswer the following question:\n{question}"

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model; you can also try "text-davinci-003"
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def local_question_answering(df, question):
    question = question.lower()

    if "mean" in question:
        for column in df.columns:
            if column.lower() in question:
                mean_value = df[column].mean()
                return f"The mean of {column} is {mean_value:.2f}."
        return "Column not found. Please specify a valid column name."

    elif "median" in question:
        for column in df.columns:
            if column.lower() in question:
                median_value = df[column].median()
                return f"The median of {column} is {median_value:.2f}."
        return "Column not found. Please specify a valid column name."

    elif "mode" in question:
        for column in df.columns:
            if column.lower() in question:
                mode_value = df[column].mode().values[0]
                return f"The mode of {column} is {mode_value:.2f}."
        return "Column not found. Please specify a valid column name."

    elif "std" in question or "standard deviation" in question:
        for column in df.columns:
            if column.lower() in question:
                std_dev_value = df[column].std()
                return f"The standard deviation of {column} is {std_dev_value:.2f}."
        return "Column not found. Please specify a valid column name."

    else:
        return None  # Indicates that the question wasn't recognized as a simple statistical question


def main(csv_file_path):
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Display basic information about the dataset
    print("Dataset Information:")
    print(df.info())
    print("Basic Statistical Summary:")
    print(df.describe())

    # Perform statistical operations on each numerical column
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        mean, median, mode, std_dev = calculate_statistics(df, column)
        # Generate insights using LLM
        stats_summary = f"Mean: {mean}, Median: {median}, Mode: {mode}, Std Dev: {std_dev}"
        generate_insights(stats_summary)

    # Plot all histograms at once
    plot_all_histograms(df)

    # Enter an interactive loop for asking questions to the LLM
    while True:
        user_question = input("Ask a question about the dataset (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break

        # First try to answer the question locally
        local_response = local_question_answering(df, user_question)
        if local_response:
            print(f"Response: {local_response}")
        else:
            # If the question can't be answered locally, fallback to the LLM
            response = ask_llm_question(df, user_question)
            print(f"LLM Response: {response}")


if __name__ == "__main__":
    import sys  # Import sys here

    if len(sys.argv) < 2:
        print("Usage: python script_name.py path/to/your_dataset.csv")
    else:
        csv_file_path = sys.argv[1]
        main(csv_file_path)
