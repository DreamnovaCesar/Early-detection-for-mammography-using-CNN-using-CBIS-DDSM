from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']

dic = classification_report(y_true, y_pred, target_names=target_names, output_dict = True)

print(dic)
print(dic['class 0'])

    