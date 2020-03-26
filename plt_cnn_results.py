import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 12})
import numpy as np

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height.
    
    Inputs:
    rects: matplotlib BarContainer object
    ax: axis object
    """
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
def plot_overall():
    '''
    Plots ShallowCNN performance on different data augmentation methods
    '''
    data_augment_labels = ['None','Subsampling','Cropped Sequential','Cropped Random', 'CWT*']
    test_accs_augment = [0.6,0.42,0.68,0.58, 0.35]

    x = np.arange(len(data_augment_labels))

    width=0.3
    fig1, ax = plt.subplots()
    rects1 = ax.bar(x, test_accs_augment, width, label='Acc')
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(data_augment_labels,rotation=15)
    ax.set_title('CNN Test Acc vs. Data Augmentation Technique')
    ax.set_ylim([0,0.8])
    autolabel(rects1,ax)

    #fig.tight_layout()
    #plt.show()

def plot_subject():
    '''
    Plots Shallow CNN performance per subject with cropped sequential
    data augmentation
    '''
    subject_acc = [0.78,0.6,0.88,0.68,0.72,0.53,0.84,0.78,0.70]
    subject_labels = np.arange(9)

    x = np.arange(len(subject_labels))

    width = 0.3
    fig2,ax = plt.subplots()
    rects1 = ax.bar(x,subject_acc, width, label='Acc')
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_labels)
    ax.set_title('Model Accuracies per subject')
    ax.set_ylim([0,1])
    autolabel(rects1,ax)

def plot_artificial_data():
    '''
    Plots the boxplots with ratios of artificial data appended
    '''
    zero_append = [0.3510,0.40,0.30]
    quarter_append = [0.3837,0.4449,0.3959]
    half_append = [0.4,0.3592,0.351]
    full_append = [0.302,0.3388,0.4041]

    data = [zero_append,quarter_append,half_append,full_append]
    fig3,ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels([r'{}%'.format(i) for i in [0,25,50,100]])
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracies vs. Percentage of Artificial Data appended')
    ax.set_xlabel('Percentage of artificial data appended')

def main():
    plot_overall()
    plot_subject()
    plot_artificial_data()
    plt.show()
if __name__ == "__main__":
    main()