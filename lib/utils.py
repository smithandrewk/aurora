def class_count(df):
    from numpy import bincount
    p,s,w = bincount(df['Class'])
    total = p + s + w
    print('Examples:\n    Total: {}\n    P: {} ({:.2f}% of total)\n    S: {} ({:.2f}% of total)\n    W: {} ({:.2f}% of total)\n'.format(
        total, p, 100 * p / total,s,100 * s / total,w,100 * w / total))
    return p,s,w