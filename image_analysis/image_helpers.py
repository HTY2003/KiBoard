import numpy as np

def graytorgb(mask, framergb):
    frame = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return frame * frame * framergb

def closest_node(node, nodes, ymin):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    closest = nodes[np.argmin(dist_2)]
    return tuple(closest)

def angle(start, centre, end):
    ba = start - centre
    bc = end - centre
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

#wrapper function for thresholding
