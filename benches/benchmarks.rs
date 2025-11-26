use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fake::{
    Fake, Faker,
    faker::{
        address::en::CityName,
        company::en::CompanyName,
        internet::en::{DomainSuffix, Username},
    },
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use xtri::{RadixTree, SearchMode};

fn benchmark_insertion(c: &mut Criterion) {
    c.bench_function("insertion_1000_random_words", |b| {
        b.iter(|| {
            let mut tree: RadixTree<usize> = RadixTree::new();
            let mut rng = StdRng::seed_from_u64(42);
            for i in 0..1000 {
                let key: String = Faker.fake_with_rng(&mut rng);
                tree.insert(black_box(&key), black_box(i));
            }
            tree
        })
    });

    c.bench_function("insertion_10000_random_words", |b| {
        b.iter(|| {
            let mut tree: RadixTree<usize> = RadixTree::new();
            let mut rng = StdRng::seed_from_u64(42);
            for i in 0..10000 {
                let key: String = Faker.fake_with_rng(&mut rng);
                tree.insert(black_box(&key), black_box(i));
            }
            tree
        })
    });

    c.bench_function("insertion_mixed_realistic_keys", |b| {
        b.iter(|| {
            let mut tree: RadixTree<usize> = RadixTree::new();
            let mut rng = StdRng::seed_from_u64(42);
            for i in 0..1000 {
                let key: String = Faker.fake_with_rng(&mut rng);
                tree.insert(black_box(&key), black_box(i));
            }
            tree
        })
    });
}

fn benchmark_search_iter(c: &mut Criterion) {
    // Setup tree with realistic data
    let mut tree: RadixTree<usize> = RadixTree::new();

    // Add various types of realistic keys
    for i in 0..5000 {
        let username: String = Username().fake();
        tree.insert(&username, i);
    }

    for i in 0..2000 {
        let company: String = CompanyName().fake();
        tree.insert(&company, i + 5000);
    }

    for i in 0..1000 {
        let city: String = CityName().fake();
        tree.insert(&city, i + 7000);
    }

    for i in 0..500 {
        let domain: String = DomainSuffix().fake();
        tree.insert(&domain, i + 8000);
    }

    // Common prefixes for search benchmarks
    let common_prefixes = ["a", "an", "the", "con", "pro", "app", "web", "com"];

    c.bench_function("search_iter_common_prefixes", |b| {
        let mut rng = StdRng::seed_from_u64(123);
        b.iter(|| {
            let prefix = common_prefixes[rng.gen_range(0..common_prefixes.len())];
            let results: Vec<_> = tree
                .search_iter(black_box(prefix), SearchMode::Prefix)
                .collect();
            black_box(results)
        })
    });

    c.bench_function("search_iter_single_char", |b| {
        let mut rng = StdRng::seed_from_u64(456);
        b.iter(|| {
            let ch = (b'a' + rng.gen_range(0..26)) as char;
            let prefix = ch.to_string();
            let results: Vec<_> = tree
                .search_iter(black_box(&prefix), SearchMode::Prefix)
                .collect();
            black_box(results)
        })
    });

    c.bench_function("search_iter_all_entries", |b| {
        b.iter(|| {
            let results: Vec<_> = tree
                .search_iter(black_box(""), SearchMode::Prefix)
                .collect();
            black_box(results)
        })
    });

    c.bench_function("search_iter_no_matches", |b| {
        b.iter(|| {
            let results: Vec<_> = tree
                .search_iter(black_box("xyz999nonexistent"), SearchMode::Prefix)
                .collect();
            black_box(results)
        })
    });

    c.bench_function("search_iter_partial_consumption", |b| {
        b.iter(|| {
            let results: Vec<_> = tree
                .search_iter(black_box("a"), SearchMode::Prefix)
                .take(10)
                .collect();
            black_box(results)
        })
    });
}

fn benchmark_merge(c: &mut Criterion) {
    c.bench_function("merge_two_1000_trees", |b| {
        b.iter_batched(
            || {
                // Setup: Create two trees with 1000 items each
                let mut tree1 = RadixTree::new();
                let mut tree2 = RadixTree::new();

                for i in 0..1000 {
                    tree1.insert(&format!("a_{:08}", i), i);
                    tree2.insert(&format!("z_{:08}", i), i + 1000);
                }

                (tree1, tree2)
            },
            |(tree1, tree2)| {
                black_box(tree1.merge(tree2, |_, v2| v2))
            },
            criterion::BatchSize::SmallInput
        )
    });

    c.bench_function("merge_overlapping_500_trees", |b| {
        b.iter_batched(
            || {
                // Setup: Create two trees with overlapping keys
                let mut tree1 = RadixTree::new();
                let mut tree2 = RadixTree::new();

                for i in 0..500 {
                    tree1.insert(&format!("key_{:08}", i), i);
                    tree2.insert(&format!("key_{:08}", i + 250), i + 1000);
                }

                (tree1, tree2)
            },
            |(tree1, tree2)| {
                black_box(tree1.merge(tree2, |v1, _| v1))
            },
            criterion::BatchSize::SmallInput
        )
    });
}

#[cfg(feature = "parallel")]
fn benchmark_parallel_build(c: &mut Criterion) {
    use criterion::BenchmarkId;

    // Compare sequential vs parallel for different sizes
    let sizes = vec![1000, 5000, 10000];

    for size in sizes {
        let mut group = c.benchmark_group("sorted_insertion");

        // Sequential insertion
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &size,
            |b, &size| {
                let items: Vec<_> = (0..size)
                    .map(|i| (format!("key_{:08}", i), i))
                    .collect();

                b.iter(|| {
                    let mut tree = RadixTree::new();
                    for (k, v) in &items {
                        tree.insert(k, *v);
                    }
                    black_box(tree)
                })
            }
        );

        // Parallel build with default chunk size
        group.bench_with_input(
            BenchmarkId::new("parallel_default", size),
            &size,
            |b, &size| {
                let items: Vec<_> = (0..size)
                    .map(|i| (format!("key_{:08}", i), i))
                    .collect();

                b.iter(|| {
                    black_box(RadixTree::from_sorted_parallel(items.clone(), None))
                })
            }
        );

        // Parallel build with 500 chunk size
        group.bench_with_input(
            BenchmarkId::new("parallel_chunk_500", size),
            &size,
            |b, &size| {
                let items: Vec<_> = (0..size)
                    .map(|i| (format!("key_{:08}", i), i))
                    .collect();

                b.iter(|| {
                    black_box(RadixTree::from_sorted_parallel(items.clone(), Some(500)))
                })
            }
        );

        // Parallel build with 2000 chunk size
        group.bench_with_input(
            BenchmarkId::new("parallel_chunk_2000", size),
            &size,
            |b, &size| {
                let items: Vec<_> = (0..size)
                    .map(|i| (format!("key_{:08}", i), i))
                    .collect();

                b.iter(|| {
                    black_box(RadixTree::from_sorted_parallel(items.clone(), Some(2000)))
                })
            }
        );

        group.finish();
    }
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    benchmark_insertion,
    benchmark_search_iter,
    benchmark_merge,
    benchmark_parallel_build
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    benchmark_insertion,
    benchmark_search_iter,
    benchmark_merge
);

criterion_main!(benches);
