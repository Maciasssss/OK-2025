# zad5.py
import sys
import time


class SuffixTreeNode:
    def __init__(self, start, end_ptr, is_leaf_suffix_start=None):
        self.start = start
        self.end_ptr = end_ptr
        self.children = {}
        self.suffix_link = None
        self.is_leaf_suffix_start = is_leaf_suffix_start

    def edge_length(self, current_global_end):
        end_val = current_global_end if isinstance(
            self.end_ptr, list) else self.end_ptr
        return end_val - self.start


class SuffixTree:
    def __init__(self, text):
        self.text = text
        self.N = len(text)
        self.root = SuffixTreeNode(-1, [-1])

        self.active_node = self.root
        self.active_edge_char_idx = -1
        self.active_length = 0

        self.remaining_suffixes = 0
        self.global_end = [-1]

        for i in range(self.N):
            self._extend(i)

    def _extend(self, i):
        self.global_end[0] = i
        self.remaining_suffixes += 1
        last_new_internal_node = None

        while self.remaining_suffixes > 0:
            if self.active_length == 0:
                self.active_edge_char_idx = i

            current_char_on_active_edge = self.text[self.active_edge_char_idx]

            if current_char_on_active_edge not in self.active_node.children:
                leaf_suffix_start = (i - self.remaining_suffixes) + 1
                new_leaf = SuffixTreeNode(
                    start=i, end_ptr=self.global_end, is_leaf_suffix_start=leaf_suffix_start)
                self.active_node.children[current_char_on_active_edge] = new_leaf

                if last_new_internal_node is not None:
                    last_new_internal_node.suffix_link = self.active_node
                    last_new_internal_node = None
            else:
                next_node = self.active_node.children[current_char_on_active_edge]
                edge_len = next_node.edge_length(self.global_end[0])

                if self.active_length >= edge_len:
                    self.active_node = next_node
                    self.active_length -= edge_len
                    self.active_edge_char_idx += edge_len
                    continue

                char_on_edge_at_active_length = self.text[next_node.start +
                                                          self.active_length]
                if char_on_edge_at_active_length == self.text[i]:
                    self.active_length += 1
                    if last_new_internal_node is not None:
                        last_new_internal_node.suffix_link = self.active_node
                        last_new_internal_node = None
                    break
                else:
                    split_node = SuffixTreeNode(
                        start=next_node.start, end_ptr=next_node.start + self.active_length)
                    self.active_node.children[current_char_on_active_edge] = split_node

                    leaf_suffix_start = (i - self.remaining_suffixes) + 1
                    new_leaf = SuffixTreeNode(
                        start=i, end_ptr=self.global_end, is_leaf_suffix_start=leaf_suffix_start)
                    split_node.children[self.text[i]] = new_leaf

                    next_node.start += self.active_length
                    split_node.children[char_on_edge_at_active_length] = next_node

                    if last_new_internal_node is not None:
                        last_new_internal_node.suffix_link = split_node
                    last_new_internal_node = split_node

            self.remaining_suffixes -= 1

            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge_char_idx = (i - self.remaining_suffixes) + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link

    def _dfs_iterative_find_smallest_rotation_start(self, original_s_len):
        stack = [(self.root, iter(sorted(self.root.children.keys())))]

        while stack:
            current_node, child_key_iterator = stack[-1]

            if current_node.is_leaf_suffix_start is not None:
                if current_node.is_leaf_suffix_start < original_s_len:
                    return current_node.is_leaf_suffix_start

            try:
                next_child_key = next(child_key_iterator)
                next_child_node = current_node.children[next_child_key]
                stack.append((next_child_node, iter(
                    sorted(next_child_node.children.keys()))))
            except StopIteration:
                stack.pop()

        return -1

    def find_lexicographically_smallest_rotation(self, original_s_len):
        start_index_in_T = self._dfs_iterative_find_smallest_rotation_start(
            original_s_len)

        if start_index_in_T != -1:
            return self.text[start_index_in_T: start_index_in_T + original_s_len]
        else:
            if original_s_len == 0:
                return ""
            return "Error: No valid rotation found"


if __name__ == '__main__':
    overall_start_time = time.perf_counter()

    input_read_start_time = time.perf_counter()
    s = sys.stdin.readline().strip()
    input_read_end_time = time.perf_counter()

    s_len = len(s)

    MAX_LEN = 5 * 10**5
    if s_len > MAX_LEN:
        print(
            f"Error: Input string length {s_len} exceeds the maximum allowed length of {int(MAX_LEN)}.", file=sys.stderr)
        sys.exit(1)

    computation_start_time = time.perf_counter()
    result = ""
    if s_len == 0:
        result = ""
    elif s_len == 1:
        result = s
    else:
        terminal_char = chr(1)

        if terminal_char in s:
            print(
                f"Error: Terminal character '{terminal_char}' found in input string. Choose a different terminal.", file=sys.stderr)
            sys.exit(1)

        text_for_tree = s + s + terminal_char

        st = SuffixTree(text_for_tree)
        result = st.find_lexicographically_smallest_rotation(s_len)

    computation_end_time = time.perf_counter()

    output_write_start_time = time.perf_counter()
    print(result)
    output_write_end_time = time.perf_counter()

    overall_end_time = time.perf_counter()

    print(f"\n--- Timing Report (seconds) ---", file=sys.stderr)
    print(
        f"Input Reading:   {input_read_end_time - input_read_start_time:.6f}", file=sys.stderr)
    print(
        f"Computation:     {computation_end_time - computation_start_time:.6f}", file=sys.stderr)
    print(
        f"Output Writing:  {output_write_end_time - output_write_start_time:.6f}", file=sys.stderr)
    print(
        f"Total Execution: {overall_end_time - overall_start_time:.6f}", file=sys.stderr)
    print(f"-----------------------------", file=sys.stderr)
