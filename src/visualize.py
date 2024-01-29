import matplotlib.pyplot as plt
import numpy as np

def visualize(predictions, test_y, test_x, labels) -> None:
    sicket_idx = np.argsort(np.sum(test_y, 1) < 1)
    fig, m_axis = plt.subplots(40, 20, figsize=(16, 32))
    for (idx, c) in zip(sicket_idx, m_axis.flatten()):
        c.imshow(test_x[idx, :, :,0], cmap="bone")
        stat_str = [n_class[:6] for n_class, n_score in zip(labels, 
                                                                  test_y[idx]) 
                             if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(labels, 
                                                                  test_y[idx], predictions[idx]) 
                             if (n_score>0.5) or (p_score>0.5)]
    c.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c.axis('off')
    fig.savefig('trained_img_predictions122.png')
    