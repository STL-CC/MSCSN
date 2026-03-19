from pipeline import (
    stage1_initialize_run,
    stage2_load_data,
    stage3_prepare_representation_file,
    stage4_discover_shapelets,
    stage5_extract_features,
    stage6_train_classifier,
)
from utils import parse_arguments

def main():
    parser = parse_arguments()
    args = parser.parse_args()

    writer = stage1_initialize_run(args)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = stage2_load_data(
        args
    )

    try:
        hdf5_path = stage3_prepare_representation_file(args, train_data, train_labels)
        final_shapelets, shapelet_dims = stage4_discover_shapelets(args, hdf5_path)
    except Exception:
        print("Error: Failed in Stage 1/Stage 2 preparation. Exiting.")
        writer.close()
        return

    if not final_shapelets:
        print("Error: No shapelets discovered. Exiting.")
        writer.close()
        return

    train_features, val_features, test_features = stage5_extract_features(
        train_data,
        val_data,
        test_data,
        final_shapelets,
        shapelet_dims,
    )

    args.input_dim = train_features.shape[1]
    if args.input_dim == 0:
        print("Error: Input dimension is 0. Exiting.")
        writer.close()
        return

    _ = stage6_train_classifier(
        args,
        writer,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    )

    writer.close()

if __name__ == "__main__":
    main()
