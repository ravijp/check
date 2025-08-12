class FeatureNormalizerV3:
    def __init__(self, spark: SparkSession, mapping: Dict[str, Dict] = RAW_TO_NORMALIZED_MAPPING):
        self.spark = spark
        self.mapping = mapping
        self.missing_value_lit = MISSING_VALUE_LIT
        self.orig_column_suffix = RAW_COLUMN_SUFFIX

    def normalize(self, df: DataFrame) -> DataFrame:
        logger.info(f"Starting normalization for {len(self.mapping)} columns")
        
        # Get columns once
        df_columns_set = set(df.columns)
        filtered_mapping = {
            col_name: rules 
            for col_name, rules in self.mapping.items() 
            if col_name in df_columns_set
        }
        
        if not filtered_mapping:
            logger.info("No columns to normalize found in dataframe")
            return df
        
        # Batch rename operations
        rename_exprs = []
        transform_exprs = []
        
        for col_name, rules in filtered_mapping.items():
            raw_col = f"{col_name}{self.orig_column_suffix}"
            keep_vals = rules.get("keep", [])
            drop_vals = rules.get("drop", [])
            
            # Build transformation expression
            if keep_vals:
                expr = F.when(F.col(raw_col).isin(keep_vals), F.col(raw_col)).otherwise(F.lit(self.missing_value_lit))
            elif drop_vals:
                expr = F.when(F.col(raw_col).isin(drop_vals), F.lit(self.missing_value_lit)).otherwise(F.col(raw_col))
            else:
                expr = F.col(raw_col)
            
            rename_exprs.append(F.col(col_name).alias(raw_col))
            transform_exprs.append(expr.alias(col_name))
        
        # Apply all renames at once
        other_cols = [F.col(c) for c in df.columns if c not in filtered_mapping]
        df = df.select(*(other_cols + rename_exprs))
        
        # Apply all transformations at once
        final_cols = [F.col(c) for c in df.columns if c not in filtered_mapping] + transform_exprs
        df = df.select(*final_cols)
        
        logger.info(f"Completed normalization for {len(filtered_mapping)} columns")
        return df
