class Node:
    def __init__(self,val,next_ = None):
        self.val = val
        self.next = next_
        
list_ = []       
def is_Panlindrome(node : Node) -> bool:
    
    if node is None:
        return True
    
    current_node = node
    while current_node is not None:
        list_.append(current_node.val)
        current_node = current_node.next
    print(list_)
    return list_[:] == list_[::-1]

if __name__ == "__main__":
    hea_node = Node(1,Node(2,Node(3,Node(2,Node(1)))))
    print(is_Panlindrome(hea_node))
