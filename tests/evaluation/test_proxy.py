import unittest

from tacorank.evaluation.proxy import build_internal_proxy, split_validation_indices


class ProxyTests(unittest.TestCase):
    def test_validation_split_is_user_disjoint_and_stable(self):
        users = ["a", "a", "b", "c", "c", "d"]
        first = split_validation_indices(users)
        second = split_validation_indices(users)
        self.assertEqual(first, second)
        a_users = {users[index] for index in first[0]}
        b_users = {users[index] for index in first[1]}
        self.assertFalse(a_users.intersection(b_users))

    def test_internal_proxy_obeys_dates_and_user_cap(self):
        rows = [
            {"date": 20220414, "user_id": "a", "row": 0},
            {"date": 20220415, "user_id": "a", "row": 1},
            {"date": 20220416, "user_id": "a", "row": 2},
            {"date": 20220417, "user_id": "a", "row": 3},
            {"date": 20220416, "user_id": "b", "row": 4},
        ]
        selected = build_internal_proxy(rows, 20220415, 20220421, impressions_per_user=2)
        self.assertEqual(sum(row["user_id"] == "a" for row in selected), 2)
        self.assertTrue(all(row["date"] >= 20220415 for row in selected))


if __name__ == "__main__":
    unittest.main()
