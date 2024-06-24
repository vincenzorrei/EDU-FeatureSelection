import matplotlib.pyplot as plt


def standard_box_plot(X):
    # Boxplot
    plt.figure(figsize=(15, 10))
    X.boxplot()
    plt.xticks(rotation=90)
    plt.title("Boxplot")
    plt.show()
