import matplotlib.pyplot as plt
import string

def plot_count(count):
    y = count
    # x = list(string.ascii_uppercase)
    x = range(26)
    plt.bar(x, y)
    plt.xticks(x, list(string.ascii_uppercase))
    plt.xlabel("characters")
    plt.ylabel("data count")
    plt.show()

if __name__ == "__main__":
    count = [789 , 766,736 ,805 ,768,775,773,734,755, 747 ,739,761,792,783 ,753 ,803 ,783 ,758 ,748 ,796 ,813 ,764,752 ,787 ,786 , 734]
    plot_count(count)