def __create_dataframe_with_neighbors(data, sampleWindow, traceWindow, inlineWindow):
    # Get the dimensions of the input array
    a, b, c = data.shape

    # Create empty lists to store the data for the DataFrame
    neighbors = []

    # Iterate over each point in the input array
    for i in range(a):
        for j in range(b):
            for k in range(c):
                # Store the point coordinates

                # Initialize lists to store the neighbors' values
                x_vals = []
                y_vals = []
                z_vals = []

                # Iterate over the neighbors' indices in each axis
                for x_offset in range(-sampleWindow, sampleWindow+1):
                  if x_offset == 0:
                    continue

                  x_idx = i + x_offset
                  if x_idx < 0 or x_idx >= a:
                    x_idx = 0 if x_idx < 0 else a - 1

                  x_vals.append(data[x_idx, j, k])


                for y_offset in range(-traceWindow, traceWindow+1):
                  if y_offset == 0:
                    continue

                  y_idx = j + y_offset
                  if y_idx < 0 or y_idx >= b:
                    y_idx = 0 if y_idx < 0 else b - 1

                  y_vals.append(data[i, y_idx, k])

                for z_offset in range(-inlineWindow, inlineWindow+1):
                  if z_offset == 0:
                    continue

                  z_idx = k + z_offset
                  if z_idx < 0 or z_idx >= c:
                    z_idx = 0 if z_idx < 0 else c - 1

                  z_vals.append(data[i, j, z_idx])

                # Combine the neighbor values into a single list
                neighbor_values = [data[i, j, k]] + x_vals + y_vals + z_vals


                # Store the neighbor values
                neighbors.append(neighbor_values)

    df = pd.DataFrame(neighbors)

    return df