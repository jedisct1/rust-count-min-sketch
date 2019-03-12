use rand;

use rand::RngCore;
use std::borrow::Borrow;
use std::cmp::max;
use std::hash::{Hash, Hasher};

use siphasher::sip::SipHasher13;
type FastHasher = SipHasher13;

use std::marker::PhantomData;
use std::mem;

macro_rules! cms_define {
    ($CountMinSketch:ident, $Counter:ty) => {
        pub struct $CountMinSketch<K> {
            counters: Vec<Vec<$Counter>>,
            offsets: Vec<usize>,
            hashers: [FastHasher; 2],
            mask: usize,
            k_num: usize,
            reset_idx: usize,
            phantom_k: PhantomData<K>,
        }

        impl<K> $CountMinSketch<K>
        where
            K: Hash,
        {
            pub fn new(
                capacity: usize,
                probability: f64,
                tolerance: f64,
            ) -> Result<Self, &'static str> {
                let width = Self::optimal_width(capacity, tolerance);
                let k_num = Self::optimal_k_num(probability);
                let counters: Vec<Vec<$Counter>> = vec![vec![0; width]; k_num];
                let offsets = vec![0; k_num];
                let hashers = [Self::sip_new(), Self::sip_new()];
                let cms = $CountMinSketch {
                    counters: counters,
                    offsets: offsets,
                    hashers: hashers,
                    mask: Self::mask(width),
                    k_num: k_num,
                    reset_idx: 0,
                    phantom_k: PhantomData,
                };
                Ok(cms)
            }

            pub fn add<Q: ?Sized>(&mut self, key: &Q, value: $Counter)
            where
                Q: Hash,
                K: Borrow<Q>,
            {
                let mut hashes = [0u64, 0u64];
                let lowest = (0..self.k_num)
                    .map(|k_i| {
                        let offset = self.offset(&mut hashes, key, k_i);
                        self.offsets[k_i] = offset;
                        self.counters[k_i][offset]
                    })
                    .min()
                    .unwrap();
                for k_i in 0..self.k_num {
                    let offset = self.offsets[k_i];
                    if self.counters[k_i][offset] == lowest {
                        self.counters[k_i][offset] =
                            self.counters[k_i][offset].saturating_add(value);
                    }
                }
            }

            pub fn increment<Q: ?Sized>(&mut self, key: &Q)
            where
                Q: Hash,
                K: Borrow<Q>,
            {
                self.add(key, 1)
            }

            pub fn estimate<Q: ?Sized>(&self, key: &Q) -> $Counter
            where
                Q: Hash,
                K: Borrow<Q>,
            {
                let mut hashes = [0u64, 0u64];
                (0..self.k_num)
                    .map(|k_i| {
                        let offset = self.offset(&mut hashes, key, k_i);
                        self.counters[k_i][offset]
                    })
                    .min()
                    .unwrap()
            }

            pub fn estimate_memory(
                capacity: usize,
                probability: f64,
                tolerance: f64,
            ) -> Result<usize, &'static str> {
                let width = Self::optimal_width(capacity, tolerance);
                let k_num = Self::optimal_k_num(probability);
                Ok(width * mem::size_of::<$Counter>() * k_num)
            }

            pub fn clear(&mut self) {
                for k_i in 0..self.k_num {
                    for counter in &mut self.counters[k_i] {
                        *counter = 0
                    }
                }
                self.reset_idx = 0;
                self.hashers = [Self::sip_new(), Self::sip_new()];
            }

            pub fn reset(&mut self) {
                for k_i in 0..self.k_num {
                    for counter in &mut self.counters[k_i] {
                        *counter /= 2;
                    }
                }
                self.reset_idx = 0;
            }

            pub fn reset_next(&mut self) -> Option<usize> {
                let idx = self.reset_idx;
                for k_i in 0..self.k_num {
                    self.counters[k_i][idx] /= 2
                }
                let next = idx.wrapping_add(1) & self.mask;
                self.reset_idx = next;
                if next != 0 {
                    Some(next)
                } else {
                    None
                }
            }

            fn optimal_width(capacity: usize, tolerance: f64) -> usize {
                let e = tolerance / (capacity as f64);
                let width = (2.0 / e).round() as usize;
                max(2, width)
                    .checked_next_power_of_two()
                    .expect("Width would be way too large")
            }

            fn mask(width: usize) -> usize {
                assert!(width > 1);
                assert_eq!(width & (width - 1), 0);
                width - 1
            }

            fn optimal_k_num(probability: f64) -> usize {
                max(1, ((1.0 - probability).ln() / 0.5f64.ln()) as usize)
            }

            fn sip_new() -> FastHasher {
                let mut rng = rand::thread_rng();
                FastHasher::new_with_keys(rng.next_u64(), rng.next_u64())
            }

            fn offset<Q: ?Sized>(&self, hashes: &mut [u64; 2], key: &Q, k_i: usize) -> usize
            where
                Q: Hash,
                K: Borrow<Q>,
            {
                if k_i < 2 {
                    let sip = &mut self.hashers[k_i as usize].clone();
                    key.hash(sip);
                    let hash = sip.finish();
                    hashes[k_i as usize] = hash;
                    hash as usize & self.mask
                } else {
                    hashes[0]
                        .wrapping_add((k_i as u64).wrapping_mul(hashes[1]) % 0xffffffffffffffc5)
                        as usize
                        & self.mask
                }
            }
        }
    };
} // macro_rules! cms_define

cms_define!(CountMinSketch8, u8);
cms_define!(CountMinSketch16, u16);
cms_define!(CountMinSketch32, u32);
cms_define!(CountMinSketch64, u64);

#[cfg(test)]
mod tests {
    #[test]
    fn test_overflow() {
        use crate::CountMinSketch8;

        let mut cms = CountMinSketch8::<&str>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..300 {
            cms.increment("key");
        }
        assert_eq!(cms.estimate("key"), u8::max_value());
    }

    #[test]
    fn test_increment() {
        use crate::CountMinSketch16;

        let mut cms = CountMinSketch16::<&str>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..300 {
            cms.increment("key");
        }
        assert_eq!(cms.estimate("key"), 300);
    }

    #[test]
    fn test_increment_multi() {
        use crate::CountMinSketch64;

        let mut cms = CountMinSketch64::<u64>::new(100, 0.99, 2.0).unwrap();
        for i in 0..1_000_000 {
            cms.increment(&(i % 100));
        }
        for key in 0..100 {
            assert!(cms.estimate(&key) >= 9_000);
        }
        cms.reset();
        for key in 0..100 {
            assert!(cms.estimate(&key) < 11_000);
        }
    }
}
