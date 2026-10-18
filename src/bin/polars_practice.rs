use polars::{
    io::SerReader,
    prelude::{CsvParseOptions, CsvReadOptions},
};

fn main() {
    println!("Hello, Polars");
    let housing_data = CsvReadOptions::default()
        .with_has_header(true)
        // .with_parse_options(CsvParseOptions::default().with_eol_char(8))
        .try_into_reader_with_file_path(Some("datasets/housing.csv".into()))
        .expect("Unable to read file path")
        .finish()
        .expect("Unable to create file");

    println!("{}", housing_data);
}
