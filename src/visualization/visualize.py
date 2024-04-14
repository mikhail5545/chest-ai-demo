import matplotlib.pyplot as plt
import numpy as np


def visualize(predictions, test_y, test_x, labels) -> None:
  """
  This function is used for visualize the information obtained after training.
  """
    sickest_idx = np.argsort(np.sum(test_y, 1)<1)
    fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
    for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
        c_ax.imshow(test_x[idx, :,:,0], cmap = 'bone')
        stat_str = [n_class[:6] for n_class, n_score in zip(labels, 
                                                                    test_y[idx]) 
                                if n_score>0.5]
        pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(labels, 
                                                                    test_y[idx], predictions[idx]) 
                                if (n_score>0.5) or (p_score>0.5)]
        c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
        c_ax.axis('off')
    fig.savefig('trained_img_predictions00.png')
