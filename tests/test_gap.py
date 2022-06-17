import unittest
from sms.gap import find_z

class TestSMSGap(unittest.TestCase):
  def test_find_z(self):
    # https://twitter.com/sup39x1207/status/1460922537769009153
    self.assertEqual(find_z(1302.07495, ((-6000, -33900), (6200, 32700))), (5962.1465, 5962.1470))

if __name__ == '__main__':
  unittest.main()
