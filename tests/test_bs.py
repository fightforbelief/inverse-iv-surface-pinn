"""
Unit tests for Black-Scholes pricing module.

Tests cover:
- Basic pricing correctness
- Edge cases (T=0, sigma=0, extreme strikes)
- Put-call parity
- Greeks accuracy
- Vectorization support
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivpinn.bs import (
    bs_call_price, bs_put_price, bs_vega, 
    bs_delta_call, bs_delta_put, bs_gamma,
    verify_put_call_parity
)


class TestBlackScholesBasics(unittest.TestCase):
    """Test basic Black-Scholes pricing functionality."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05
        self.q = 0.02
    
    def test_atm_call_put_parity(self):
        """Test that ATM call and put have specific relationship."""
        C = bs_call_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        P = bs_put_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        
        # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
        expected = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        assert_allclose(C - P, expected, rtol=1e-10)
    
    def test_call_price_positive(self):
        """Call price should always be non-negative."""
        C = bs_call_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(C, 0.0)
    
    def test_put_price_positive(self):
        """Put price should always be non-negative."""
        P = bs_put_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(P, 0.0)
    
    def test_call_intrinsic_value(self):
        """Call price should be at least intrinsic value."""
        C = bs_call_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        intrinsic = max(self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T), 0)
        self.assertGreaterEqual(C, intrinsic - 1e-10)
    
    def test_put_intrinsic_value(self):
        """Put price should be at least intrinsic value."""
        P = bs_put_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        intrinsic = max(self.K * np.exp(-self.r * self.T) - self.S * np.exp(-self.q * self.T), 0)
        self.assertGreaterEqual(P, intrinsic - 1e-10)


class TestBlackScholesEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_expiration_deep_itm_call(self):
        """At expiration, deep ITM call should equal intrinsic value."""
        S, K = 110.0, 100.0
        C = bs_call_price(S, K, T=0.0, sigma=0.2, r=0.0, q=0.0)
        assert_allclose(C, S - K, rtol=1e-10)
    
    def test_expiration_deep_otm_call(self):
        """At expiration, deep OTM call should be worthless."""
        S, K = 90.0, 100.0
        C = bs_call_price(S, K, T=0.0, sigma=0.2, r=0.0, q=0.0)
        assert_allclose(C, 0.0, atol=1e-10)
    
    def test_expiration_deep_itm_put(self):
        """At expiration, deep ITM put should equal intrinsic value."""
        S, K = 90.0, 100.0
        P = bs_put_price(S, K, T=0.0, sigma=0.2, r=0.0, q=0.0)
        assert_allclose(P, K - S, rtol=1e-10)
    
    def test_expiration_deep_otm_put(self):
        """At expiration, deep OTM put should be worthless."""
        S, K = 110.0, 100.0
        P = bs_put_price(S, K, T=0.0, sigma=0.2, r=0.0, q=0.0)
        assert_allclose(P, 0.0, atol=1e-10)
    
    def test_zero_volatility_itm_call(self):
        """With zero volatility, ITM call should be discounted intrinsic."""
        S, K, T = 110.0, 100.0, 1.0
        r, q = 0.05, 0.0
        C = bs_call_price(S, K, T, sigma=0.0, r=r, q=q)
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert_allclose(C, expected, rtol=1e-6)
    
    def test_zero_volatility_otm_call(self):
        """With zero volatility, OTM call should be worthless."""
        S, K = 90.0, 100.0
        C = bs_call_price(S, K, T=1.0, sigma=0.0, r=0.0, q=0.0)
        assert_allclose(C, 0.0, atol=1e-10)
    
    def test_very_high_strike(self):
        """Call with very high strike should be near zero."""
        C = bs_call_price(100.0, 1000.0, 1.0, 0.2, 0.0, 0.0)
        self.assertLess(C, 1e-6)
    
    def test_very_low_strike(self):
        """Call with very low strike should be close to spot."""
        S = 100.0
        C = bs_call_price(S, 1.0, 1.0, 0.2, 0.0, 0.0)
        assert_allclose(C, S, rtol=0.01)  # Within 1%
    
    def test_put_call_parity_general(self):
        """Test put-call parity with general parameters."""
        test_cases = [
            (100, 100, 1.0, 0.2, 0.05, 0.02),
            (100, 90, 0.5, 0.3, 0.03, 0.01),
            (100, 110, 2.0, 0.15, 0.04, 0.0),
        ]
        
        for S, K, T, sigma, r, q in test_cases:
            with self.subTest(S=S, K=K, T=T):
                self.assertTrue(verify_put_call_parity(S, K, T, sigma, r, q))


class TestBlackScholesGreeks(unittest.TestCase):
    """Test Greeks calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05
        self.q = 0.02
    
    def test_vega_positive(self):
        """Vega should always be non-negative."""
        vega = bs_vega(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(vega, 0.0)
    
    def test_vega_zero_at_expiration(self):
        """Vega should be zero at expiration."""
        vega = bs_vega(self.S, self.K, T=0.0, sigma=self.sigma, r=self.r, q=self.q)
        assert_allclose(vega, 0.0, atol=1e-10)
    
    def test_vega_symmetric_call_put(self):
        """Vega should be the same for call and put with same parameters."""
        # Vega is derived from the same d1 term, so it's identical for calls and puts
        vega = bs_vega(self.S, self.K, self.T, self.sigma, self.r, self.q)
        
        # Verify by numerical differentiation
        epsilon = 1e-6
        C1 = bs_call_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        C2 = bs_call_price(self.S, self.K, self.T, self.sigma + epsilon, self.r, self.q)
        vega_numerical_call = (C2 - C1) / epsilon
        
        P1 = bs_put_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        P2 = bs_put_price(self.S, self.K, self.T, self.sigma + epsilon, self.r, self.q)
        vega_numerical_put = (P2 - P1) / epsilon
        
        assert_allclose(vega, vega_numerical_call, rtol=1e-4)
        assert_allclose(vega, vega_numerical_put, rtol=1e-4)
    
    def test_vega_numerical_consistency(self):
        """Test that vega matches numerical differentiation."""
        epsilon = 1e-6
        vega_analytical = bs_vega(self.S, self.K, self.T, self.sigma, self.r, self.q)
        
        C1 = bs_call_price(self.S, self.K, self.T, self.sigma - epsilon, self.r, self.q)
        C2 = bs_call_price(self.S, self.K, self.T, self.sigma + epsilon, self.r, self.q)
        vega_numerical = (C2 - C1) / (2 * epsilon)
        
        assert_allclose(vega_analytical, vega_numerical, rtol=1e-5)
    
    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        delta = bs_delta_call(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
    
    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        delta = bs_delta_put(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(delta, -1.0)
        self.assertLessEqual(delta, 0.0)
    
    def test_delta_call_put_relationship(self):
        """Delta_call - Delta_put should equal exp(-qT)."""
        delta_c = bs_delta_call(self.S, self.K, self.T, self.sigma, self.r, self.q)
        delta_p = bs_delta_put(self.S, self.K, self.T, self.sigma, self.r, self.q)
        expected = np.exp(-self.q * self.T)
        assert_allclose(delta_c - delta_p, expected, rtol=1e-10)
    
    def test_gamma_positive(self):
        """Gamma should always be non-negative."""
        gamma = bs_gamma(self.S, self.K, self.T, self.sigma, self.r, self.q)
        self.assertGreaterEqual(gamma, 0.0)
    
    def test_gamma_zero_at_expiration(self):
        """Gamma should be zero at expiration."""
        gamma = bs_gamma(self.S, self.K, T=0.0, sigma=self.sigma, r=self.r, q=self.q)
        assert_allclose(gamma, 0.0, atol=1e-10)
    
    def test_gamma_numerical_consistency(self):
        """Test that gamma matches numerical second derivative."""
        epsilon = 1e-4
        gamma_analytical = bs_gamma(self.S, self.K, self.T, self.sigma, self.r, self.q)
        
        C1 = bs_call_price(self.S - epsilon, self.K, self.T, self.sigma, self.r, self.q)
        C2 = bs_call_price(self.S, self.K, self.T, self.sigma, self.r, self.q)
        C3 = bs_call_price(self.S + epsilon, self.K, self.T, self.sigma, self.r, self.q)
        gamma_numerical = (C3 - 2*C2 + C1) / (epsilon**2)
        
        assert_allclose(gamma_analytical, gamma_numerical, rtol=1e-3)


class TestBlackScholesVectorization(unittest.TestCase):
    """Test that functions work with vectorized inputs."""
    
    def test_call_price_vectorized(self):
        """Test call pricing with array inputs."""
        S = np.array([90, 100, 110])
        K = 100
        T = 1.0
        sigma = 0.2
        
        prices = bs_call_price(S, K, T, sigma)
        
        # Should return array of same shape as S
        self.assertEqual(prices.shape, S.shape)
        
        # All prices should be positive
        self.assertTrue(np.all(prices >= 0))
        
        # Higher spot should give higher call price
        self.assertLess(prices[0], prices[1])
        self.assertLess(prices[1], prices[2])
    
    def test_put_price_vectorized(self):
        """Test put pricing with array inputs."""
        S = 100
        K = np.array([90, 100, 110])
        T = 1.0
        sigma = 0.2
        
        prices = bs_put_price(S, K, T, sigma)
        
        # Should return array of same shape as K
        self.assertEqual(prices.shape, K.shape)
        
        # All prices should be positive
        self.assertTrue(np.all(prices >= 0))
        
        # Higher strike should give higher put price
        self.assertLess(prices[0], prices[1])
        self.assertLess(prices[1], prices[2])
    
    def test_vega_vectorized(self):
        """Test vega with array inputs."""
        S = 100
        K = 100
        T = np.array([0.25, 0.5, 1.0, 2.0])
        sigma = 0.2
        
        vegas = bs_vega(S, K, T, sigma)
        
        # Should return array of same shape as T
        self.assertEqual(vegas.shape, T.shape)
        
        # All vegas should be non-negative
        self.assertTrue(np.all(vegas >= 0))
    
    def test_multidimensional_arrays(self):
        """Test with 2D arrays."""
        S = np.array([[90, 100], [100, 110]])
        K = 100
        T = 1.0
        sigma = 0.2
        
        prices = bs_call_price(S, K, T, sigma)
        
        # Should maintain shape
        self.assertEqual(prices.shape, S.shape)
        
        # Test monotonicity
        self.assertLess(prices[0, 0], prices[0, 1])
        self.assertLess(prices[1, 0], prices[1, 1])
    
    def test_broadcasting(self):
        """Test numpy broadcasting behavior."""
        S = np.array([90, 100, 110])[:, None]  # Shape (3, 1)
        K = np.array([95, 100, 105])  # Shape (3,)
        T = 1.0
        sigma = 0.2
        
        prices = bs_call_price(S, K, T, sigma)
        
        # Should broadcast to (3, 3)
        self.assertEqual(prices.shape, (3, 3))


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling."""
    
    def test_negative_spot_raises(self):
        """Negative spot price should raise ValueError."""
        with self.assertRaises(ValueError):
            bs_call_price(-100, 100, 1.0, 0.2)
    
    def test_negative_strike_raises(self):
        """Negative strike price should raise ValueError."""
        with self.assertRaises(ValueError):
            bs_call_price(100, -100, 1.0, 0.2)
    
    def test_negative_time_raises(self):
        """Negative time to maturity should raise ValueError."""
        with self.assertRaises(ValueError):
            bs_call_price(100, 100, -1.0, 0.2)
    
    def test_negative_volatility_raises(self):
        """Negative volatility should raise ValueError."""
        with self.assertRaises(ValueError):
            bs_call_price(100, 100, 1.0, -0.2)
    
    def test_zero_inputs_valid(self):
        """Zero values for K, T, sigma should be handled (not raise errors)."""
        # These are edge cases but mathematically valid
        try:
            bs_call_price(100, 100, 0.0, 0.2)  # T=0 is expiration
            bs_call_price(100, 100, 1.0, 0.0)  # sigma=0 is deterministic
        except Exception as e:
            self.fail(f"Zero inputs raised unexpected exception: {e}")


class TestKnownValues(unittest.TestCase):
    """Test against known numerical values from external sources."""
    
    def test_case_1_atm(self):
        """Test ATM option with known value."""
        # Parameters from Hull's "Options, Futures, and Other Derivatives"
        # Example: S=100, K=100, T=1, sigma=0.2, r=0.05, q=0
        C = bs_call_price(100, 100, 1.0, 0.2, r=0.05, q=0.0)
        # Expected value approximately 10.45
        assert_allclose(C, 10.4506, rtol=1e-3)
    
    def test_case_2_itm_call(self):
        """Test ITM call with known value."""
        # S=110, K=100, T=0.5, sigma=0.25, r=0.05, q=0.02
        C = bs_call_price(110, 100, 0.5, 0.25, r=0.05, q=0.02)
        # Should be greater than intrinsic value
        intrinsic = 110 * np.exp(-0.02 * 0.5) - 100 * np.exp(-0.05 * 0.5)
        self.assertGreater(C, intrinsic)
        # Approximate value: 13.7
        assert_allclose(C, 13.7, rtol=0.05)
    
    def test_case_3_otm_put(self):
        """Test OTM put with known value."""
        # S=100, K=90, T=0.25, sigma=0.3, r=0.04, q=0.01
        P = bs_put_price(100, 90, 0.25, 0.3, r=0.04, q=0.01)
        # Should be positive but small
        self.assertGreater(P, 0)
        self.assertLess(P, 5)
        # Approximate value: 1.9
        assert_allclose(P, 1.9, rtol=0.1)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBlackScholesBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackScholesEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackScholesGreeks))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackScholesVectorization))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestKnownValues))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests(verbosity=2)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)