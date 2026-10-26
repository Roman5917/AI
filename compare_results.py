"""Aggregate only 20 validated real runs; sample SD across five seeds."""
import common as c
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def format_number(value, digits=None):
    text = f'{value:g}' if digits is None else f'{value:.{digits}f}'
    return text.replace('.', ',').replace('-', '−')


def format_numeric_axes(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: format_number(value)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: format_number(value)))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def confusion_plot(cm, labels, title, path, fractional=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    threshold = cm.max() / 2
    for i in range(c.NUM_CLASSES):
        for j in range(c.NUM_CLASSES):
            text = format_number(cm[i,j], 1) if fractional else str(int(cm[i,j]))
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                    color='white' if cm[i,j] > threshold else '#132238')
    ax.set_xticks(range(c.NUM_CLASSES), labels, rotation=40, ha='right')
    ax.set_yticks(range(c.NUM_CLASSES), labels)
    ax.set(xlabel='Передбачений клас', ylabel='Справжній клас', title=title)
    colorbar = fig.colorbar(im, ax=ax, label='Середня кількість спостережень' if fractional else 'Кількість спостережень')
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: format_number(value)))
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    data = c.load_data()
    all_rows, summaries = [], []
    plots = c.ROOT / 'figures'; plots.mkdir(exist_ok=True)
    labels = data['class_names'].tolist()
    for name in c.FRAMEWORKS:
        rows = [c.validate_run(name, seed, data) for seed in c.SEEDS]
        all_rows.extend(rows)
        summary = dict(framework=name, runs=len(rows), version=rows[0]['version'])
        for field in c.METRIC_FIELDS:
            values = np.asarray([r[field] for r in rows])
            summary[field + '_mean'] = float(values.mean())
            summary[field + '_sd'] = float(values.std(ddof=1))
        summaries.append(summary)
        curves = np.array([np.loadtxt(c.run_path(name, seed) / 'learning_curve.csv',
                                     delimiter=',', skiprows=1)[:, 1:] for seed in c.SEEDS])
        curve_rows = []
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        epochs = np.arange(1, c.EPOCHS+1)
        for idx, (label, color) in enumerate((('Навчання', '#176b87'), ('Валідація', '#c27524'))):
            mean, sd = curves[:, :, idx].mean(axis=0), curves[:, :, idx].std(axis=0, ddof=1)
            ax.plot(epochs, mean, label=label, color=color)
            ax.fill_between(epochs, mean-sd, mean+sd, color=color, alpha=.18)
        for epoch in range(c.EPOCHS):
            curve_rows.append(dict(epoch=epoch+1, train_ce_mean=float(curves[:, epoch, 0].mean()),
                                   train_ce_sd=float(curves[:, epoch, 0].std(ddof=1)),
                                   validation_ce_mean=float(curves[:, epoch, 1].mean()),
                                   validation_ce_sd=float(curves[:, epoch, 1].std(ddof=1))))
        write_csv(c.OUTPUT_DIR / f'{name}_learning_curve_mean_sd.csv', curve_rows)
        ax.set(xlabel='Епоха', ylabel='Перехресна ентропія, нат',
               title=f'{name}: навчальні та валідаційні втрати\nСереднє ± вибіркове СВ за 5 запусками')
        format_numeric_axes(ax)
        ax.legend(fontsize=9); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(plots / f'{name}_learning_curves.png', dpi=300); plt.close(fig)
        mean_cm = np.mean([r['confusion_matrix'] for r in rows], axis=0)
        np.savetxt(c.OUTPUT_DIR / f'{name}_confusion_mean.csv', mean_cm, delimiter=',', fmt='%.1f')
        confusion_plot(mean_cm, labels, f'{name}: середня матриця помилок\nза 5 запусками',
                       plots / f'{name}_confusion_matrix.png', fractional=True)
        for seed, row in zip(c.SEEDS, rows):
            out = c.run_path(name, seed)
            confusion_plot(np.asarray(row['confusion_matrix']), labels, f'{name}: запуск із початковим значенням {seed}', out / 'confusion_matrix.png')
            curve = np.loadtxt(out / 'learning_curve.csv', delimiter=',', skiprows=1)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(curve[:, 0], curve[:, 1], label='Навчання'); ax.plot(curve[:, 0], curve[:, 2], label='Валідація')
            ax.set(xlabel='Епоха', ylabel='Перехресна ентропія, нат', title=f'{name}: початкове значення {seed}')
            format_numeric_axes(ax)
            ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=300); plt.close(fig)
    if len({r['dataset_sha256'] for r in all_rows}) != 1:
        raise ValueError('Dataset differs across runs')
    write_csv(c.OUTPUT_DIR / 'all_runs.csv',
              [{key: r[key] for key in ('framework', 'version', 'seed', *c.METRIC_FIELDS)} for r in all_rows])
    write_csv(c.OUTPUT_DIR / 'comparison_mean_sd.csv', summaries)
    formatted = [dict(framework=s['framework'], **{field: f'{format_number(s[field+"_mean"], 6)} ± {format_number(s[field+"_sd"], 6)}'
                                                  for field in c.METRIC_FIELDS}) for s in summaries]
    write_csv(c.OUTPUT_DIR / 'comparison_formatted.csv', formatted)
    colors = ['#176b87', '#629e91', '#c79647', '#9876aa']
    for field, title, ylabel, scale in (
            ('accuracy', 'Точність класифікації на тестовій вибірці', 'Точність, %', 100),
            ('f1_macro', 'Macro-F1 на тестовій вибірці', 'Macro-F1, %', 100),
            ('train_seconds', 'Час навчання за 100 епох', 'Час навчання, с', 1)):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        mean = np.array([s[field+'_mean'] for s in summaries]) * scale
        sd = np.array([s[field+'_sd'] for s in summaries]) * scale
        bars = ax.bar(list(c.FRAMEWORKS), mean, yerr=sd, capsize=6, color=colors)
        for bar, m, s in zip(bars, mean, sd):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+max(mean)*.025,
                    f'{format_number(m, 2)} ± {format_number(s, 2)}', ha='center', fontsize=9)
        ax.set(xlabel='Фреймворк', ylabel=ylabel, title=f'{title}\nСереднє ± вибіркове СВ, n=5')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: format_number(value)))
        ax.set_ylim(0, max(mean+sd)*1.18); ax.grid(axis='y', alpha=.2)
        fig.tight_layout(); fig.savefig(plots / f'comparison_{field}.png', dpi=300); plt.close(fig)
    print('Verified 20 real runs. Mean ± sample SD (ddof=1):')
    for s in summaries:
        print(f'{s["framework"]:14s} Accuracy={s["accuracy_mean"]:.5f} ± {s["accuracy_sd"]:.5f}; '
              f'F1={s["f1_macro_mean"]:.5f} ± {s["f1_macro_sd"]:.5f}; '
              f'train={s["train_seconds_mean"]:.2f} ± {s["train_seconds_sd"]:.2f} s')


if __name__ == '__main__':
    main()
