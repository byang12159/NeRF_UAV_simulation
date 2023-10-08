class TreeNode:
    def __init__(self, value=0):
        self.value = value
        self.children = []

def get_paths(node, current_path=[], all_paths=[]):
    """
    Recursive function to get all paths from root to leaves.
    """
    if not node:
        return []

    # Append the current node's value to the current path.
    current_path.append(node.value)

    # If it's a leaf node, append the current path to all_paths.
    if not node.children:
        all_paths.append(list(current_path))
    else:
        # Otherwise, recursively search through the children.
        for child in node.children:
            get_paths(child, current_path, all_paths)

    # Backtrack: remove the current node's value to explore other paths.
    current_path.pop()

    return all_paths

# Example usage
if __name__ == "__main__":
    # Constructing a sample tree
    #        1
    #      / | \
    #     2  3  4
    #    /|   \
    #   5 6    7
    root = TreeNode(1)
    child2 = TreeNode(2)
    child3 = TreeNode(3)
    child4 = TreeNode(4)

    root.children = [child2, child3, child4]
    child2.children = [TreeNode(5), TreeNode(6)]
    child3.children = [TreeNode(7)]

    paths = get_paths(root)
    for path in paths:
        print(path)