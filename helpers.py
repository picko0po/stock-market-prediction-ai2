def train(x_df, y_series, epochs=100):

    x_tensor = torch.tensor(
        x_df.values,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_series.values,
        dtype=torch.float32
    )

    dataset = LSTMStocksDataset(
        x_tensor,
        y_tensor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = LSTMStocksModule()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    loss_function = torch.nn.HuberLoss(
        delta=0.02
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x, y in dataloader:

            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_function(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Loss: {total_loss / len(dataloader):.6f}"
            )

    return model
