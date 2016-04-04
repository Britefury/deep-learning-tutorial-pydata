def coerce_number_parameter(x, num_type=int, name='x'):
    if isinstance(x, num_type):
        return x
    elif isinstance(x, (str, unicode)):
        return num_type(x)
    else:
        raise TypeError('Parameter {0} should either be of type str or {1}'.format(name, num_type))

def coerce_num_list_parameter(x, N=None, num_type=int, name='x'):
    if isinstance(x, list):
        if N is not None and len(x) != N:
            if len(x) == 1:
                x = x * N
            else:
                raise ValueError('parameter {0} should have {1} items, not {2}'.format(name, N, len(x)))
        return [coerce_number_parameter(p, num_type=num_type, name='<item of ' + name + '>') for p in x]
    elif isinstance(x, num_type):
        return coerce_num_list_parameter([x], N=N, num_type=num_type, name=name)
    elif isinstance(x, (str, unicode)):
        if ',' in x:
            items = [i.strip() for i in x.split(',')]
            items = [i for i in items if i != '']
            return coerce_num_list_parameter(items, N=N, num_type=num_type, name=name)
        else:
            x = num_type(x)
            if N is None:
                return [x]
            else:
                return [x] * N


import unittest

class Test_parameter_coercion (unittest.TestCase):
    def test_coerce_number_parameter(self):
        self.assertEqual(coerce_number_parameter(1, int, 'x'), 1)
        self.assertRaises(TypeError, lambda: coerce_number_parameter(1, float, 'x'))
        self.assertEqual(coerce_number_parameter(1.0, float, 'x'), 1.0)
        self.assertEqual(coerce_number_parameter('1', int, 'x'), 1)
        self.assertEqual(coerce_number_parameter('1', float, 'x'), 1.0)

    def test_coerce_num_list_parameter(self):
        # Pass list through unchanged
        self.assertEqual(coerce_num_list_parameter([1, 2, 3], 3, int, 'x'), [1, 2, 3])
        # Without length constraint
        self.assertEqual(coerce_num_list_parameter([1, 2, 3], None, int, 'x'), [1, 2, 3])
        # Violate length constraint
        self.assertRaises(ValueError, lambda: coerce_num_list_parameter([1, 2, 3], 4, int, 'x'))
        # Repeat single item list to match length constraint
        self.assertEqual(coerce_num_list_parameter([1], 3, int, 'x'), [1, 1, 1])
        # Single item list with no length constraint
        self.assertEqual(coerce_num_list_parameter([1], None, int, 'x'), [1])

        # List of strings
        self.assertEqual(coerce_num_list_parameter(['1', '2', '3'], 3, int, 'x'), [1, 2, 3])
        self.assertEqual(coerce_num_list_parameter(['1', '2', '3'], None, int, 'x'), [1, 2, 3])
        self.assertRaises(ValueError, lambda: coerce_num_list_parameter(['1', '2', '3'], 4, int, 'x'))
        self.assertEqual(coerce_num_list_parameter(['1'], 3, int, 'x'), [1, 1, 1])
        self.assertEqual(coerce_num_list_parameter(['1'], None, int, 'x'), [1])

        # Number
        self.assertEqual(coerce_num_list_parameter(1, 3, int, 'x'), [1, 1, 1])
        self.assertEqual(coerce_num_list_parameter(1, None, int, 'x'), [1])

        # String
        self.assertEqual(coerce_num_list_parameter('1', 3, int, 'x'), [1, 1, 1])
        self.assertEqual(coerce_num_list_parameter('1', None, int, 'x'), [1])

        # String of numbers
        # Pass list through unchanged
        self.assertEqual(coerce_num_list_parameter('1, 2, 3', 3, int, 'x'), [1, 2, 3])
        self.assertEqual(coerce_num_list_parameter('1, 2, 3,', 3, int, 'x'), [1, 2, 3])
        self.assertEqual(coerce_num_list_parameter('1, 2, 3', None, int, 'x'), [1, 2, 3])
        self.assertEqual(coerce_num_list_parameter('1, 2, 3,', None, int, 'x'), [1, 2, 3])
        self.assertRaises(ValueError, lambda: coerce_num_list_parameter('1, 2, 3', 4, int, 'x'))
        self.assertEqual(coerce_num_list_parameter('1,', 3, int, 'x'), [1, 1, 1])
        self.assertEqual(coerce_num_list_parameter('1,', None, int, 'x'), [1])
