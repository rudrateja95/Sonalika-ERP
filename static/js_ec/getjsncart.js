function getOrderData() {

    const orders = [];

    document.querySelectorAll(".order-item").forEach(item => {

        orders.push({

            style_no: item.querySelector('[data-role="style_no"]').value,

            qty: item.querySelector('[data-role="qty"]').value,

            diamond_clarity: item.querySelector('[data-role="diamond_clarity"]').value,

            diamond_color: item.querySelector('[data-role="diamond_color"]').value,

            gold_color: item.querySelector('[data-role="gold_color"]').value,

            gold_purity: item.querySelector('[data-role="gold_purity"]').value,

            remarks: item.querySelector('[data-role="remarks"]').value

        });

    });

    return orders;

}