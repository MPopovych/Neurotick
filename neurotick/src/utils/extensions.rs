use std::hash::Hash;

use indexmap::IndexSet;

pub trait Distinct<T, B: Hash + Eq> {
    fn distinct_vec<F>(self, f: F) -> Vec<B>
    where
        Self: Sized,
        F: (FnMut(T) -> B);
}

impl<T, B: Eq + Hash, U: Iterator<Item = T>> Distinct<T, B> for U {
    fn distinct_vec<F>(self, f: F) -> Vec<B>
    where
        Self: Sized,
        F: (FnMut(T) -> B),
    {
        let set = self.map(f).collect::<IndexSet<B>>();
        return set.into_iter().collect::<Vec<_>>();
    }
}

pub fn eq_vecs<T: PartialEq>(a: &Vec<T>, b: &Vec<T>) -> bool {
    let matching = a.iter().zip(b.iter()).filter(|&(a, b)| a == b).count();
    matching == a.len() && matching == b.len()
}

#[cfg(test)]
pub mod test {
    use super::Distinct;

    #[test]
    fn test_distinct() {
        let vec_original = vec![2, 3, 4, 4, 6, 7, 8, 1, 2];
        let vec_distinct = vec_original.iter().distinct_vec(|f| *f);
        assert_eq!(vec_distinct, vec![2, 3, 4, 6, 7, 8, 1])
    }
}