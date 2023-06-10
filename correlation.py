import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer

df = pd.read_csv('merged_file_Hm0.csv')

plot_variable = 'Hm0'


df_values = df.iloc[:, 1:]

def correlations(df):
    corr_matrix = df_values.corr()

    corr_matrix.to_csv('correlations.csv')

    # Select correlation values greater than 0.7
    high_corr_values = corr_matrix[corr_matrix > 0.7].dropna(how='all', axis=1)

    # Print high correlation values
    print(high_corr_values)

    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('corr_matrix.png')
    plt.show()
    return corr_matrix



print(correlations(df))


def plot_variable(df, variable):
    for values in df_values:
        if values == 'WR10' or values == 'PQFF10':
            continue
        plt.plot(df_values[values].iloc[:144], label=values)

    # Add axis labels and a title
    plt.xlabel('Time')
    plt.ylabel('Significant wave height')
    plt.title(f'Plot of {plot_variable}')
    plt.legend()

    plt.savefig(f'Plots/{plot_variable}.png')
    # Display the plot
    plt.show()



