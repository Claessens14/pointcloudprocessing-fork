

# Development Notes

This is how they call the plot function
```
https://github.com/QingyongHu/RandLA-Net/blob/3926be6658221172b680d38d475551fda6d589b5/utils/6_fold_cv.py#L34
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            xyzrgb = np.concatenate([points, colors], axis=-1)
            Plot.draw_pc(xyzrgb)  # visualize raw point clouds
            Plot.draw_pc_sem_ins(points, labels)  # visualize ground-truth
            Plot.draw_pc_sem_ins(points, pred)  # visualize prediction
```





