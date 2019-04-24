
class MetricDisplay:

    def СonfusionMatrix(self, y, y_pred, title='Confusion matrix', cmap=plt.cm.Blues, size=(8,8)):
        cm = confusion_matrix(y_test, y_pred) 
        
        plt.figure(figsize=size)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(iris.target_names))
        plt.xticks(tick_marks, iris.target_names, rotation=45)
        plt.yticks(tick_marks, iris.target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
