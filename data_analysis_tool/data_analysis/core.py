cols = self.df.columns.tolist()
        n = len(cols)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                c1, c2 = cols[i], cols[j]
                is_num1 = np.issubdtype(self.df[c1].dtype, np.number)
                is_num2 = np.issubdtype(self.df[c2].dtype, np.number)

                valid_data = self.df[[c1, c2]].dropna()
                if valid_data.empty or valid_data[c1].nunique() <= 1 or valid_data[c2].nunique() <= 1:
                    matrix[i, j] = 0.0
                    continue

                v1, v2 = valid_data[c1], valid_data[c2]

                if is_num1 and is_num2:
                    matrix[i, j] = v1.corr(v2, method='pearson') if len(v1) > 1 and v1.std() > 0 and v2.std() > 0 else 0.0
                elif not is_num1 and not is_num2:
                    confusion_matrix = pd.crosstab(v1, v2)
                    n_obs = confusion_matrix.sum().sum()
                    r, k = confusion_matrix.shape

                    if n_obs > 0 and min(r - 1, k - 1) > 0:
                        try:
                            chi2 = stats.chi2_contingency(confusion_matrix)[0]
                            matrix[i, j] = np.sqrt(chi2 / (n_obs * min(r - 1, k - 1)))
                        except ValueError:
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0
                else:
                    num_col, cat_col = (v1, v2) if is_num1 else (v2, v1)
                    groups = [group.values for _, group in num_col.groupby(cat_col) if len(group) > 0]

                    if len(groups) > 1 and sum(len(g) for g in groups) > len(groups):
                        try:
                            f_val, _ = stats.f_oneway(*groups)
                            n_total = len(num_col)
                            k_groups = len(groups)
                            denominator = f_val * (k_groups - 1) + (n_total - k_groups)
                            matrix[i, j] = np.sqrt((f_val * (k_groups - 1)) / denominator) if denominator > 0 else 0.0
                        except ValueError:
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0

        matrix = np.nan_to_num(matrix)
        fig = px.imshow(matrix, x=cols, y=cols, color_continuous_scale='RdBu_r', zmin=-1.0, zmax=1.0,
                        title="Unified Association Heatmap (Pearson, Cramer's V, Eta)")
        fig.show()
