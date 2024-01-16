import matplotlib.pyplot as plt


def autolabel(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='black',
        )


def print_plot(list_values, list_labels, title, x_label, y_label, color, horizontal_line_value=None):
    plt.figure(figsize=(13, 9))

    # Customize the bar chart appearance
    bars = plt.bar(list_labels, list_values, color=color, edgecolor='black', linewidth=1.2)

    # Add data labels on top of the bars
    mini = 0
    maxi = 1
    plt.ylim(mini, maxi)  # Adjust the y-axis limits for better visualization
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=14)

    # Customize the horizontal line appearance
    if horizontal_line_value is not None:
        horizontal_line_value = max(list_values)
        plt.axhline(y=horizontal_line_value, color='red', linestyle='--', linewidth=2,
                    label=f'Horizontal Line at ~ {horizontal_line_value:.2f}')
        plt.legend()

    # Customize the overall layout
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    autolabel(bars, list_values)

    plt.show()


def pie_chart(error_percent, title):
    other_percent = 100 - error_percent

    # Labels for the two parts of the pie chart
    labels = ['Errors', 'Correct Classified']

    # Different colors for the two parts of the pie chart
    colors = ['#FF5733', '#33FF57']  # Replace these with your desired colors

    # Data to plot
    sizes = [error_percent, other_percent]

    # Plot the pie chart
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

    # Add a title to the pie chart
    plt.title(title, fontsize=14)

    # Equal aspect ratio ensures that the pie is drawn as a circle
    plt.axis('equal')

    # Display the pie chart
    plt.show()

list_labels = ['Before', 'After']
list_labels_n_algo = ['RandomSearch', 'GridSearch']
list_time_for_hyper_algo = [12.015170097351074, 72.54500985145569]
list_time_for_hyper_rf = [29.964808225631714, 613.4225769042969]
scores_algo = [0.71764064171123, 0.7363329768270945]
scores_algo_rf = [0.7926340898976413, 0.8030440587449933]

list_values_ab_time_hyper = [0.1915536642074585, 0.5773224115371705]
list_values_rf_time_hyper = [0.7287919044494628, 11.296740198135376]
list_values_ab_accuracy_testing = [0.678462770216173, 0.7231385108086469]
list_values_rf_accuracy_testing = [0.800160128102482, 0.8118494795836669]

# print_plot(list_values_ab_time_hyper, list_labels, 'Time took on training before and after hyperparameters tunning',
#            'AdaBoost', 'Time', 'skyblue')

# print_plot(list_time_for_hyper_rf, list_labels_n_algo, 'Time took for hyperparameters tunning (RandomForest)',
#            "Algorithms", "Time(in seconds)", "skyblue")
# print_plot(scores_algo_rf, list_labels_n_algo, 'Best Score for hyperparameters tunning (RandomForest)',
#            "Algorithms", "Best Score", "skyblue", horizontal_line_value=1.0)

# print_plot(list_values_rf_time_hyper, list_labels, 'Time took on training before and after hyperparameters tunning',
#            'RandomForest', 'Time', 'forestgreen')

print_plot(list_values_rf_accuracy_testing, list_labels, 'Accuracy testing before and after hyperparameters tunning',
           'RandomForest', 'Accuracy', 'forestgreen', horizontal_line_value=1.0)