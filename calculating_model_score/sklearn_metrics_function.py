from sklearn import metrics

y_test = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
y_predict = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]

def show_metrics(y_test, y_predict, labels):
    print('准确率:', metrics.accuracy_score(y_test, y_predict))  # 预测准确率输出

    print('宏平均精确率:', metrics.precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(y_test, y_predict, average='micro'))  # 预测微平均精确率输出
    print('加权平均精确率:', metrics.precision_score(y_test, y_predict, average='weighted'))  # 预测加权平均精确率输出

    print('宏平均召回率:', metrics.recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出
    print('微平均召回率:', metrics.recall_score(y_test, y_predict, average='micro'))  # 预测微平均召回率输出
    print('加权平均召回率:', metrics.recall_score(y_test, y_predict, average='micro'))  # 预测加权平均召回率输出

    print('宏平均F1-score:', metrics.f1_score(y_test, y_predict, labels=labels, average='macro'))  # 预测宏平均f1-score输出
    print('微平均F1-score:', metrics.f1_score(y_test, y_predict, labels=labels, average='micro'))  # 预测微平均f1-score输出
    print('加权平均F1-score:', metrics.f1_score(y_test, y_predict, labels=labels, average='weighted'))  # 预测加权平均f1-score输出

    print('混淆矩阵输出:\n', metrics.confusion_matrix(y_test, y_predict))  # 混淆矩阵输出
    print('分类报告:\n', metrics.classification_report(y_test, y_predict))  # 分类报告输出

#show_metrics(y_test=y_test, y_predict=y_predict, labels=[1,2,3,4])
def delete_both_sides_is_O_word(y_test_list, clean_y_predict_list):
    new_y_test_list, new_clean_y_predict_list = [], []
    for test, pred in zip(y_test_list, clean_y_predict_list):
        if test=="O" and pred=="O":
            continue
        new_y_test_list.append(test)
        new_clean_y_predict_list.append(pred)
    assert len(new_y_test_list)==len(new_clean_y_predict_list)
    return new_y_test_list, new_clean_y_predict_list