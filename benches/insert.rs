use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use xtri::RadixTree;

fn benchmark_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_foo");

    for size in [100, 1_000, 10_000, 20_000, 50_000, 100_000] {
        let input: Vec<String> = serde_json::from_slice(std::fs::read(format!("/Users/allevo/repos/oramacore_lib/bar_{}.json", size)).unwrap().as_slice()).unwrap();
        let input2: Vec<_> = input.iter().cloned().enumerate().map(|(v, k)| (k, v)).collect();

        group.bench_with_input(BenchmarkId::new("insert", size), &input, |bencher, input| {
            bencher.iter(|| {
                let mut tree = RadixTree::new();
                for (i, key) in input.iter().enumerate() {
                    tree.insert(black_box(key), black_box(i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("insert_ordered", size), &input2, |bencher, input| {
            bencher.iter(|| {
                RadixTree::from_sorted_parallel(
                    input.clone(),
                    None,
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_insertion
);

criterion_main!(benches);
