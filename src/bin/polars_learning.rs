use chrono::prelude::*;
use polars::prelude::*;

// use crate::round::RoundMode;
use polars::prelude::RoundMode;
// use polars_ops::series::RoundMode;
use plotly::common::Mode;
use plotly::{Plot, Scatter};

fn _line_and_scatter_plot() {
    let trace1 = Scatter::new(vec![1, 2, 3, 4], vec![10, 15, 13, 17])
        .name("trace1")
        .mode(Mode::Markers);
    let trace2 = Scatter::new(vec![2, 3, 4, 5], vec![16, 5, 11, 9])
        .name("trace2")
        .mode(Mode::Lines);
    let trace3 = Scatter::new(vec![1, 2, 3, 4], vec![12, 9, 15, 12]).name("trace3");

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.show();
}

fn main() {
    // ---[Creating DataFrame]-----------------------------------------------------------
    let df: DataFrame = df!(
        "name" => ["Alice Archer", "Ben Browni", "Chloe Cooper", "Daniel Donovan", "Kingsley Michael"],
        "birthdate" => [
            NaiveDate::from_ymd_opt(1997, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(1985, 2, 15).unwrap(),
            NaiveDate::from_ymd_opt(1983, 3, 22).unwrap(),
            NaiveDate::from_ymd_opt(1981, 4, 30).unwrap(),
            NaiveDate::from_ymd_opt(1988, 10, 3).unwrap(),
        ],
        "weight" => [57.9, 72.5, 53.6, 83.1, 79.2],  // (kg)
        "height" => [1.56, 1.77, 1.65, 1.75, 1.82],  // (m)
    ).expect("Repair flaws in dataframe design");

    // println!("{df}");

    // ---[Writing DataFrame to csv]------------------------------------------------------
    // let mut file = File::create("data/output.csv").expect("Error in file creation");
    // CsvWriter::new(&mut file)
    //     .include_header(true)
    //     .with_separator(b',')
    //     .finish(&mut df).expect("Error in writing csv");

    // ---[Reading DataFrame]-------------------------------------------------------------
    let df_csv = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some("data/output.csv".into()))
        .expect("Unable to parse csv data")
        .finish()
        .expect("Unable to Read file.");

    println!("{df_csv}");

    // ---[Expressions and Contexts.]------------------------------------------------------

    // Provide a modular and flexible way of expressing data transformations.
    // - select()
    // - with_columns()
    // - filter()
    // - group_by()

    let result_one = df
        .clone()
        .lazy()
        .select([
            col("name"),
            col("birthdate").dt().year().alias("birth_year"),
            (col("weight") / col("height").pow(2)).alias("bmi"),
        ])
        .collect()
        .expect("Calculate the values correctly.");
    println!("{result_one}");

    let result_two = df
        .clone()
        .lazy()
        .select([
            col("name"),
            (cols(["weight", "height"]).as_expr() * lit(1.00))
                .round(0, RoundMode::default())
                .name()
                .suffix("-5%"), // .name().suffix() comes as package deal.
        ])
        .collect();
    println!("{:?}", result_two);

    let result_three = df
        .clone()
        .lazy()
        .select([
            col("name"),
            (col("weight") * lit(2.3))
                .alias("Weight")
                .round(1, RoundMode::default())
                .name()
                .suffix("+4%"),
            (col("height") * lit(234.56))
                .alias("Height")
                .round(3, RoundMode::default())
                .name()
                .suffix("+55%"),
        ])
        .collect();
    println!("{:?}", result_three);

    // ? If the data comes without the header, then how do I give them header.

    // line_and_scatter_plot();

    // ---[with_columns]------------------

    let with_col = df
        .clone()
        .lazy()
        .with_columns([
            col("birthdate").dt().year().alias("birth_year"),
            (col("weight") / col("height").pow(2)).alias("bmi"),
        ])
        .collect()
        .expect("Error correcting the `with columns` function");

    println!("{with_col}");

    // ---[filter]------------------------

    let filter_fn = df
        .clone()
        .lazy()
        .filter(col("birthdate").dt().year().lt(lit(1990)))
        .collect()
        .expect("Error in the filter fn");

    println!("{filter_fn}");

    let result = df
        .clone()
        .lazy()
        .filter(
            col("birthdate")
                .is_between(
                    //this is broken, repair it.
                    lit(NaiveDate::from_ymd_opt(1982, 12, 31).unwrap()),
                    lit(NaiveDate::from_ymd_opt(1996, 1, 1).unwrap()),
                    ClosedInterval::Both,
                )
                .and(col("height").gt(lit(1.7))),
        )
        .collect()
        .expect("Error in filter function 2");
    println!("{result}");

    // ---[group_by]-----------------

    // Used to group together the rows of the dataframe that share the same value across one or
    // more expressions.

    // Use `group_by_stable` if you want the Python behaviour of `maintain_order=True`.
    let result = df
        .clone()
        .lazy()
        .group_by([(col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade")])
        .agg([len()])
        .collect()
        .expect("Error in `group_by` fn");
    println!("{result}");
}
